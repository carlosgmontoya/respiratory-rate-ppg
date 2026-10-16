"""Estimación de la frecuencia respiratoria (RR) desde la PPG, pensada para generalizar entre poblaciones.

Idea: en la PPG la respiración aparece como un pico LOCAL sobre un fondo de baja frecuencia cuya forma
cambia con la población (plano en pacientes clínicos, descendente tipo 1/f en sanos despiertos).
Por eso no se busca el máximo global del espectro, sino el pico que más sobresale del fondo.

Métodos por ventana de 30 s:
  A   FFT original (notebooks 02/03): máximo global de la RIIV filtrada, banda 6–35 rpm.
  D   Conteo de respiraciones en el tiempo con pasa-alto fijo de 0.17 Hz (ganador en CASE, notebook 12).
  S   Pico sobre el fondo: ajuste robusto log-log del fondo y pico de mayor prominencia sobre él.
  S-AR  Igual que S, pero con espectro autorregresivo (Burg), de mejor resolución en ventanas cortas.
  Da  Conteo adaptativo: pasa-alto = factor × RR de S (permite RR bajas reales).
  F   Fusión: media de S y Da si coinciden (± tolerancia); si no, la ventana no se estima (abstención).
S, D y Da se calculan sobre RIIV, RIAV y RIFV latido a latido y se toma la mediana (≥ 2 disponibles).

Uso general (cualquier sensor):
    from rr_generalizable import estimar_rr
    tabla = estimar_rr(ppg, fs)                # una fila por ventana de 30 s, paso 10 s
"""
from fractions import Fraction

import numpy as np
import pandas as pd
from scipy.signal import butter, decimate, detrend, filtfilt, find_peaks, resample_poly

FS = 125                                       # Hz, frecuencia de trabajo
FS_M = 4                                       # Hz, series latido a latido remuestreadas
VENTANA_S, PASO_S = 30, 10
N = VENTANA_S * FS

PARAMS = {                                     # fijados antes de ver resultados
    "rpm_min": 6, "rpm_max": 35,               # rango de RR fisiológica buscada
    "fondo_rpm": (3, 40),                      # rango del ajuste del fondo
    "fondo_percentil": 40,                     # puntos por debajo de este percentil del residuo → fondo
    "armonico_tol_rpm": 1.5,                   # tolerancia para considerar que hay pico en f/2
    "armonico_frac": None,                     # regla f/2 desactivada (None): con 0.5 elegía subarmónicos
                                               # espurios en BIDMC (prueba con bidmc01); la prominencia ya
                                               # favorece la fundamental. Valor numérico = activarla.
    "hp_fijo_hz": 0.17,                        # método D
    "hp_factor": 0.6,                          # método Da: corte = factor × RR(S)
    "lp_conteo_hz": 0.7,
    "tol_fusion_rpm": 2.0,                     # método F
    "lp_riiv_hz": 0.5, "lp_art_hz": 0.5,       # método A y detector de artefactos (config congelada)
    "umbral_art": 3.0,
    "min_latidos": 15,
    "ar_orden": 10,                            # método S-AR: orden del modelo autorregresivo (Burg)
}


# ----------------------------------------------------------------------------- preprocesado
def a_125hz(ppg, fs):
    """Remuestrea a 125 Hz con un filtro polifásico (fs puede ser 64, 128, 256, 1000…)."""
    ppg = np.asarray(ppg, float)
    if fs == FS:
        return ppg
    fr = Fraction(FS, int(round(fs))).limit_denominator(1000)
    return resample_poly(ppg, fr.numerator, fr.denominator)


def espectro(x, fs, nfft):
    """Espectro de potencia (frecuencia en rpm) tras destendencia y ventana de Hann."""
    x = detrend(x) * np.hanning(len(x))
    return np.fft.rfftfreq(nfft, 1 / fs) * 60, np.abs(np.fft.rfft(x, nfft)) ** 2


def latidos(ppg):
    """Picos y valles de la PPG a 125 Hz; devuelve tiempos y series latido a latido limpias.
    Limpieza: intervalo 0.33–1.5 s y a menos de 30 % de la mediana local de 11 latidos."""
    b, a = butter(2, [0.5, 8], btype="band", fs=FS)
    x = filtfilt(b, a, ppg)
    esc = 1.4826 * np.median(np.abs(x - np.median(x)))
    pk, _ = find_peaks(x, distance=int(0.33 * FS), prominence=0.5 * esc)
    if len(pk) < 3:
        return np.array([]), {k: np.array([]) for k in ("RIIV", "RIAV", "RIFV")}
    vl = np.array([p0 + np.argmin(x[p0:p1]) for p0, p1 in zip(pk[:-1], pk[1:])], int)
    pk = pk[1:]
    t = pk / FS
    ibi = np.diff(t)
    t, pk, vl = t[1:], pk[1:], vl[1:]
    med = pd.Series(ibi).rolling(11, center=True, min_periods=3).median().to_numpy()
    ok = (ibi >= 0.33) & (ibi <= 1.5) & (np.abs(ibi - med) <= 0.3 * med)
    y = ppg - np.median(ppg)
    return t[ok], {"RIIV": ((y[pk] + y[vl]) / 2)[ok], "RIAV": (y[pk] - y[vl])[ok], "RIFV": ibi[ok]}


def serie_4hz(t, v, t0, min_latidos=PARAMS["min_latidos"]):
    m = (t >= t0) & (t < t0 + VENTANA_S)
    if m.sum() < min_latidos:
        return None
    return np.interp(t0 + np.arange(VENTANA_S * FS_M) / FS_M, t[m], v[m])


def mediana3(vals):
    v = np.array(list(vals), float)
    return float(np.nanmedian(v)) if np.isfinite(v).sum() >= 2 else np.nan


def burg(x, orden):
    """Coeficientes AR por el método de Burg. Devuelve (a, sigma2) con a[0] = 1."""
    x = np.asarray(x, float)
    f = b = x.copy()
    a = np.array([1.0])
    e = np.dot(x, x) / len(x)
    for _ in range(orden):
        ff, bb = f[1:], b[:-1]
        den = np.dot(ff, ff) + np.dot(bb, bb)
        k = -2 * np.dot(ff, bb) / den if den > 0 else 0.0
        a = np.concatenate([a, [0.0]])
        a = a + k * a[::-1]
        f, b = ff + k * bb, bb + k * ff
        e *= (1 - k * k)
    return a, e


def espectro_ar(x, fs, nfft, orden):
    """Espectro autorregresivo (Burg) tras destendencia; frecuencia en rpm."""
    a, e = burg(detrend(x), orden)
    f = np.fft.rfftfreq(nfft, 1 / fs)
    H = np.fft.rfft(a, nfft)
    return f * 60, e / np.maximum(np.abs(H) ** 2, 1e-12)


# ----------------------------------------------------------------------------- métodos
def pico_sobre_fondo(x, p=PARAMS, fs=FS_M, nfft=1024, devolver_detalle=False, metodo="fft"):
    """Método S en una serie (4 Hz): ajusta el fondo log10(P) = a + b·log10(f) de forma robusta
    (ajuste, se conservan los puntos con residuo bajo el percentil indicado, reajuste) y elige el pico
    de mayor prominencia sobre el fondo dentro de [rpm_min, rpm_max]. Regla de armónicos opcional
    (`armonico_frac`): si existe un pico en f/2 con al menos esa fracción de la prominencia, se elige f/2."""
    f, P = espectro(x, fs, nfft) if metodo == "fft" else espectro_ar(x, fs, nfft, p["ar_orden"])
    lo, hi = p["fondo_rpm"]
    m = (f >= lo) & (f <= hi) & (P > 0)
    lf, lP = np.log10(f[m]), np.log10(P[m])
    coef = np.polyfit(lf, lP, 1)
    res = lP - np.polyval(coef, lf)
    base = res <= np.percentile(res, p["fondo_percentil"])
    coef = np.polyfit(lf[base], lP[base], 1)
    resid = lP - np.polyval(coef, lf)                       # prominencia sobre el fondo (décadas)
    fm = f[m]
    pk, _ = find_peaks(resid)
    pk = pk[(fm[pk] >= p["rpm_min"]) & (fm[pk] <= p["rpm_max"]) & (resid[pk] > 0)]
    if len(pk) == 0:
        return (np.nan, np.nan, None) if devolver_detalle else (np.nan, np.nan)
    k = pk[np.argmax(resid[pk])]
    rr, prom = fm[k], resid[k]
    mitad = pk[np.abs(fm[pk] - rr / 2) <= p["armonico_tol_rpm"]]
    if p.get("armonico_frac") is not None and len(mitad) and fm[mitad].min() >= p["rpm_min"]:
        k2 = mitad[np.argmax(resid[mitad])]
        if resid[k2] >= p["armonico_frac"] * prom:
            rr, prom = fm[k2], resid[k2]
    # La prominencia decide QUÉ pico; la frecuencia se toma del máximo local del espectro sin aplanar
    # (restar un fondo inclinado desplaza ligeramente el máximo del residuo).
    cerca = (fm >= rr - 1.5) & (fm <= rr + 1.5)
    rr = fm[cerca][np.argmax(P[m][cerca])]
    if devolver_detalle:
        return rr, prom, {"f": f, "P": P, "f_fondo": fm, "fondo": 10 ** np.polyval(coef, lf)}
    return rr, prom


def conteo(x, hp_hz, p=PARAMS, fs=FS_M):
    """Conteo de respiraciones en el tiempo (tipo count-orig): pasa-banda [hp, 0.7] Hz; picos por encima
    de 0.2 × Q3 de los picos con un valle < 0 entre picos consecutivos; RR = 60 / mediana del intervalo."""
    hp = float(np.clip(hp_hz, 0.03, 0.5))
    b, a = butter(2, [hp, p["lp_conteo_hz"]], btype="band", fs=fs)
    x = filtfilt(b, a, x)
    pk, _ = find_peaks(x)
    if len(pk) < 3:
        return np.nan
    pk = pk[x[pk] > 0.2 * np.percentile(x[pk], 75)]
    if len(pk) == 0:
        return np.nan
    val = [pk[0]]
    for q in pk[1:]:
        if x[val[-1]:q].min() < 0:
            val.append(q)
    if len(val) < 3:
        return np.nan
    rr = 60 / np.median(np.diff(val) / fs)
    return rr if p["rpm_min"] <= rr <= p["rpm_max"] else np.nan


def fusion(s, d, tol):
    """Método F: media si S y Da coinciden dentro de ±tol rpm; NaN (abstención) si no."""
    s, d = np.asarray(s, float), np.asarray(d, float)
    return np.where(np.abs(s - d) <= tol, (s + d) / 2, np.nan)


class Estimador:
    """Prepara una señal completa (PPG a cualquier fs) y estima la RR de cualquier ventana con `ventana(t0)`.
    factores_hp: factores de Da a calcular (para análisis de sensibilidad)."""

    def __init__(self, ppg, fs, p=PARAMS, invertir=False, factores_hp=(PARAMS["hp_factor"],)):
        self.p, self.factores = p, tuple(factores_hp)
        self.ppg = a_125hz(ppg, fs) * (-1 if invertir else 1)
        b, a = butter(4, p["lp_riiv_hz"], btype="low", fs=FS)
        self.riiv_lp = filtfilt(b, a, self.ppg)
        b, a = butter(4, p["lp_art_hz"], btype="low", fs=FS)
        self.riiv_art = filtfilt(b, a, self.ppg)
        n5 = 5 * FS
        self.tipico = np.median([np.std(self.riiv_art[i:i + n5]) for i in range(0, len(self.ppg) - n5 + 1, n5)])
        self.t, self.mods = latidos(self.ppg)
        self.duracion = len(self.ppg) / FS

    def inicios(self):
        return np.round(np.arange(0, self.duracion - VENTANA_S + 1e-9, PASO_S), 3)

    def ventana(self, t0):
        p, n5 = self.p, 5 * FS
        i0 = int(round(t0 * FS))
        out = {"Excursión": max(np.std(self.riiv_art[i:i + n5]) for i in range(i0, i0 + N - n5 + 1, n5)) / self.tipico}
        x5 = decimate(decimate(self.riiv_lp[i0:i0 + N], 5, zero_phase=True), 5, zero_phase=True)
        f, P = espectro(x5, 5, 2048)
        bnd = (f >= p["rpm_min"]) & (f <= p["rpm_max"])
        out["A"] = f[bnd][np.argmax(P[bnd])]
        series = {k: serie_4hz(self.t, v, t0, p["min_latidos"]) for k, v in self.mods.items()}
        s_mod, d_mod, ar_mod = {}, {}, {}
        for k, s in series.items():
            s_mod[k] = pico_sobre_fondo(s, p)[0] if s is not None else np.nan
            ar_mod[k] = pico_sobre_fondo(s, p, metodo="ar")[0] if s is not None else np.nan
            d_mod[k] = conteo(s, p["hp_fijo_hz"], p) if s is not None else np.nan
            out[f"S {k}"] = s_mod[k]
        out["D"] = mediana3(d_mod.values())
        out["S"] = mediana3(s_mod.values())
        out["S-AR"] = mediana3(ar_mod.values())
        for fac in self.factores:
            if np.isfinite(out["S"]):
                hp = fac * out["S"] / 60
                out[f"Da {fac}"] = mediana3([conteo(s, hp, p) if s is not None else np.nan for s in series.values()])
            else:
                out[f"Da {fac}"] = np.nan
        fac0 = p["hp_factor"] if p["hp_factor"] in self.factores else self.factores[0]
        out["Da"] = out[f"Da {fac0}"]
        out["F"] = float(fusion(out["S"], out["Da"], p["tol_fusion_rpm"]))
        out["Artefacto"] = out["Excursión"] > p["umbral_art"]
        return out


def estimar_rr(ppg, fs, p=PARAMS, invertir=False):
    """Interfaz general: RR por ventana (30 s, paso 10 s) para una PPG de cualquier sensor.
    Columnas: inicio, A, D, S, Da, F (NaN = abstención), Artefacto. Recomendado: usar F y descartar
    las ventanas con Artefacto=True."""
    est = Estimador(ppg, fs, p, invertir)
    filas = [{"Inicio (s)": t0, **est.ventana(t0)} for t0 in est.inicios()]
    cols = ["Inicio (s)", "A", "D", "S", "Da", "F", "Artefacto", "Excursión"]
    return pd.DataFrame(filas)[cols]
