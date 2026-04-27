# ------------------ 1. INTRODUCCIÓN DE DATOS ------------------
import pandas as pd
import numpy as np

# Cargar datos observados (estación) y simulados (GCM)
obs = pd.read_csv('Estacion_AeroNqn.csv', parse_dates=['fecha'], index_col='fecha')
gcm = pd.read_csv('MPI_ESM1_2HR_ssp245_AeroNqn.csv', parse_dates=['date'], index_col='date')

# Supongamos que las columnas son:
# obs['prec']   → precipitación observada (mm/día)
# gcm['pr_mm']  → precipitación simulada (mm/día)

# Definir períodos de calibración y futuro
obs_cal = obs['prec']['1990-01-01':'2010-12-31']
gcm_cal = gcm['pr_mm']['1990-01-01':'2010-12-31']
gcm_fut = gcm['pr_mm']['2040-01-01':'2060-12-31']


# ------------------ 2. DESARROLLO DEL SCRIPT ------------------
def bias_factor_escala(obs_cal, gcm_cal, gcm_fut):
    """
    Método del Factor de Escala para precipitación.
    Corrección mensual: factor = mean(obs) / mean(gcm) por mes.
    """
    factor_mensual = {}
    for mes in range(1, 13):
        obs_m = obs_cal[obs_cal.index.month == mes]
        gcm_m = gcm_cal[gcm_cal.index.month == mes]
        if len(obs_m) == 0 or len(gcm_m) == 0:
            factor_mensual[mes] = 1.0
        else:
            media_gcm = gcm_m.mean()
            if media_gcm > 0:
                factor_mensual[mes] = obs_m.mean() / media_gcm
            else:
                factor_mensual[mes] = 1.0

    # Aplicar factor al período futuro
    gcm_corr = gcm_fut.copy()
    for mes, factor in factor_mensual.items():
        mask = gcm_corr.index.month == mes
        gcm_corr.loc[mask] = gcm_corr.loc[mask] * factor

    # Clipping: precipitación no puede ser negativa
    gcm_corr = gcm_corr.clip(lower=0)
    return gcm_corr, factor_mensual


# ------------------ 3. APLICACIÓN Y EXPORTACIÓN ------------------
# Aplicar corrección
gcm_corr, factores = bias_factor_escala(obs_cal, gcm_cal, gcm_fut)

# Guardar resultados
gcm_corr.to_csv('MPI_ESM1_2HR_ssp245_AeroNqn_corr.csv')

print("Factores mensuales aplicados:", factores)
print("Serie corregida guardada en CSV.")
