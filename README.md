# Modelo AVIA - Predicción de Fragilidad en Adultos Mayores

[![AVIA](https://img.shields.io/badge/AVIA-Health%20Assessment-blue)](https://avia.cuidadosdesalud.org.ar/)

**AVIA** es un modelo de machine learning desarrollado para la evaluación y predicción de fragilidad en adultos mayores, basado en datos clínicos, funcionales y sociales.

## 📋 Tabla de Contenidos

- [Descripción](#descripción)
- [Características](#características)
- [Instalación](#instalación)
- [Uso](#uso)
- [Estructura de Datos](#estructura-de-datos)
- [Ejemplo de Uso](#ejemplo-de-uso)
- [Documentación Técnica](#documentación-técnica)
- [Colaboradores y Sponsors](#colaboradores-y-sponsors)
- [Licencia](#licencia)

## 🎯 Descripción

El modelo AVIA utiliza un pipeline de XGBoost entrenado con datos de estudios longitudinales (ELSA) para predecir el riesgo de fragilidad en adultos mayores. El modelo analiza múltiples factores incluyendo:

- **Datos demográficos**: edad, sexo, estado civil, educación, ocupación
- **Indicadores físicos**: peso, altura, IMC, circunferencia abdominal
- **Condiciones de salud**: enfermedades cardiovasculares, diabetes, EPOC, artrosis, etc.
- **Funcionalidad**: dificultades para caminar, subir escalones, equilibrio
- **Salud mental**: memoria, estado de ánimo, depresión, demencia
- **Factores sociales**: soporte social, uso de tecnología, actividad física

### Métricas del Modelo

- **ROC AUC**: 0.8466
- **Umbral de riesgo**:
  - Bajo: < 0.33
  - Medio: 0.33 - 0.66
  - Alto: ≥ 0.66

## ✨ Características

- 🎯 Predicción precisa de fragilidad en adultos mayores
- 📊 Análisis de múltiples factores de riesgo
- 🔄 Pipeline completo de preprocesamiento incluido
- 🐍 API simple y fácil de usar en Python
- 📈 Basado en evidencia científica

## 📦 Instalación

### Requisitos

- Python 3.8 o superior
- pip

### Instalación de dependencias

```bash
pip install -r requirements.txt
```

O instalar manualmente:

```bash
pip install scikit-learn==1.3.2 xgboost==2.1.1 pandas==2.2.3 joblib
```

## 🚀 Uso

### Carga del Modelo

```python
from joblib import load
import pandas as pd

# Cargar el modelo
pipeline = load('model_pipeline.pkl')
```

### Predicción

El modelo requiere datos en formato de diccionario con campos en español. Ver [Estructura de Datos](#estructura-de-datos) para más detalles.

```python
# Preparar los datos
data = {
    'edad': 75,
    'estado_civil': 'casado',
    'sexo': 'varon',
    # ... más campos (ver ejemplo completo)
}

# Convertir a DataFrame
df = pd.DataFrame([data])

# Realizar predicción
probability = pipeline.predict_proba(df)[:, 1][0]
risk_score = probability

# Interpretar el resultado
if risk_score < 0.33:
    risk_level = 'bajo'
elif risk_score < 0.66:
    risk_level = 'medio'
else:
    risk_level = 'alto'
```

## 📊 Estructura de Datos

El modelo espera un diccionario o DataFrame con las siguientes columnas (en español):

### Campos Principales

#### Demográficos
- `edad`: int - Edad en años
- `estado_civil`: str - 'soltero', 'casado', 'concubino', 'divorciado', 'viudo'
- `sexo`: str - 'varon', 'mujer'
- `escolaridad`: str - '1' (universidad), '2' (técnico), '3' (bachillerato), '4' (secundaria), '5' (primaria), '6' (otro), '7' (sin estudios)
- `ocupacion`: str - Ocupación del paciente
- `ingresos_brutos`: float - Ingresos brutos (opcional)

#### Físicos
- `peso`: float - Peso en kg
- `altura`: float - Altura en cm
- `indice_masa_corporal`: float - IMC calculado
- `obesidad_abdominal`: float - Circunferencia abdominal en cm

#### Condiciones de Salud
- `CV_HTA`: float - Hipertensión (0/1)
- `CV_stroke`: float - ACV (0/1)
- `CV_angina`: float - Angina (0/1)
- `CV_ICC`: float - Insuficiencia cardíaca (0/1)
- `diabetes`: float - Diabetes (0/1)
- `EPOC`: float - EPOC (0/1)
- `artrosis`: float - Artrosis (0/1)
- `osteoporosis`: float - Osteoporosis (0/1)
- `in_urinaria`: int - Incontinencia urinaria
- `d_mentales`: float - Desórdenes mentales (0/1)

#### Funcionalidad
- `fuma`: float - Fumador (puede ser NaN)
- `alcohol`: float - Consumo de alcohol
- `audicion`: int - Audición (1: buena, 2: regular, 3: mala)
- `vision`: int - Visión (1: buena, 2: regular, 3: mala)
- `caidas`: float - Caídas (0/1 o NaN)
- `equilibrio`: float - Problemas de equilibrio (puede ser NaN)

#### Estado de Salud
- `estado_salud`: float - Estado de salud percibido
- `dolor`: float - Nivel de dolor
- `test_silla`: float - Tiempo test de silla en segundos
- `memoria`: float - Memoria percibida
- `suenio`: float - Calidad del sueño
- `soledad`: float - Nivel de soledad

#### Actividad y Social
- `usa_internet_email`: float - Uso de internet/email (0/1)
- `tiene_celular`: float - Tiene celular (0/1)
- `soporte_social`: float - Nivel de soporte social (puede ser NaN)
- `demencia`: float - Demencia (0/1)
- `depresion`: float - Depresión (0/1, puede ser NaN)
- `actividad_fisica_1`: float - Actividad física vigorosa
- `actividad_fisica_2`: float - Actividad física moderada
- `actividad_fisica_3`: float - Actividad física ligera
- `fatigabilidad_1`: float - Fatigabilidad 1
- `fatigabilidad_2`: float - Fatigabilidad 2
- `fuerza_mano_d_promedio`: float - Fuerza de mano promedio
- `tiempo_caminar_promedio`: float - Tiempo promedio de caminata

> **Nota**: Muchos campos son opcionales y pueden ser `NaN` o `None`. El modelo está diseñado para manejar valores faltantes.

## 💡 Ejemplo de Uso

Ver el archivo [`example.py`](example.py) para un ejemplo completo de uso del modelo.

Ejemplo básico:

```python
from joblib import load
import pandas as pd

# Cargar modelo
pipeline = load('model_pipeline.pkl')

# Datos de ejemplo
data = {
    'edad': 72,
    'estado_civil': 'casado',
    'sexo': 'varon',
    'escolaridad': '4',
    'ingresos_brutos': 15000.0,
    'CV_HTA': 1.0,
    'CV_stroke': 0.0,
    'CV_angina': 0.0,
    'CV_ICC': 0.0,
    'diabetes': 1.0,
    'EPOC': 0.0,
    'artrosis': 1.0,
    'osteoporosis': 0.0,
    'in_urinaria': 2,
    'd_mentales': 0.0,
    'fuma': None,
    'alcohol': 3.0,
    'obesidad_abdominal': 105.5,
    'audicion': 2,
    'vision': 2,
    'caidas': 1.0,
    'estado_salud': 3.0,
    'dolor': 2.0,
    'equilibrio': 1.0,
    'test_silla': 8.5,
    'memoria': 4.0,
    'suenio': 2.0,
    'soledad': 2.0,
    'usa_internet_email': 0.0,
    'tiene_celular': 1.0,
    'soporte_social': 2.0,
    'demencia': 0.0,
    'depresion': 1.0,
    'indice_masa_corporal': 28.5,
    'altura': 175.0,
    'peso': 87.3,
    'actividad_fisica_1': 1.0,
    'actividad_fisica_2': 1.0,
    'actividad_fisica_3': 2.0,
    'fatigabilidad_1': 2.0,
    'fatigabilidad_2': 2.0,
    'fuerza_mano_d_promedio': 35.5,
    'tiempo_caminar_promedio': 3.2
}

# Convertir a DataFrame
df = pd.DataFrame([data])

# Realizar predicción
probability = pipeline.predict_proba(df)[:, 1][0]
risk_score = probability

print(f"Probabilidad de fragilidad: {risk_score:.4f}")

# Interpretar resultado
if risk_score < 0.33:
    risk_level = 'bajo'
    diagnosis = 'robusto'
elif risk_score < 0.66:
    risk_level = 'medio'
    diagnosis = 'pre-frágil'
else:
    risk_level = 'alto'
    diagnosis = 'frágil'

print(f"Nivel de riesgo: {risk_level}")
print(f"Diagnóstico: {diagnosis}")
```

## 📚 Documentación Técnica

### Modelo Base

- **Algoritmo**: XGBoost Classifier
- **Entrenamiento**: Random Search Cross-Validation
- **Dataset**: ELSA (English Longitudinal Study of Ageing)
- **Preprocesamiento**: Incluido en el pipeline (imputación, encoding, etc.)

### Pipeline de Preprocesamiento

El modelo incluye un pipeline completo que:
1. Maneja valores faltantes mediante imputación
2. Codifica variables categóricas
3. Normaliza variables numéricas si es necesario
4. Aplica transformaciones necesarias

### Interpretación de Resultados

- **Probabilidad < 0.33**: Bajo riesgo de fragilidad (robusto)
- **Probabilidad 0.33 - 0.66**: Riesgo moderado de fragilidad (pre-frágil)
- **Probabilidad ≥ 0.66**: Alto riesgo de fragilidad (frágil)

## 🤝 Colaboradores y Sponsors

### Reconocimientos

¡AVIA fue galardonado con el Premio Japan International Cooperation Agency (JICA) en BAILA Shibuya 2025! [Más información](https://www.linkedin.com/company/japan-international-cooperation-agency-jica-/?lipi=urn%3Ali%3Apage%3Ad_flagship3_detail_base%3BDsXucutNT6yxuj2HmaU0SQ%3D%3D)

### Instituciones Colaboradoras

Este proyecto es posible gracias al apoyo y colaboración de las siguientes instituciones:

<!-- Aquí se pueden agregar logos e información de las instituciones que apoyan el proyecto -->

- **[Universidad de La Coruña (UDC)](https://www.udc.es/)** - El proyecto comenzó y continúa con el apoyo de la Universidad de La Coruña, específicamente con el **LABIC** (Laboratorios de Innovación Ciudadana), un programa impulsado por la Secretaría General Iberoamericana (SEGIB) que promueve soluciones innovadoras para desafíos ciudadanos en América Latina, España y Portugal
- **[Cuidados de Salud](https://avia.cuidadosdesalud.org.ar/)** - Plataforma de evaluación de salud
- **[ELSA Study](https://www.elsa-project.ac.uk/)** - English Longitudinal Study of Ageing

### Equipo de Desarrollo

- Equipo de Machine Learning - Desarrollo del modelo
- Equipo de Salud - Validación clínica
- Equipo de Software - Implementación y despliegue

### Cómo Contribuir

Si desea contribuir al proyecto o patrocinar su desarrollo, por favor contacte a través del sitio web: [https://avia.cuidadosdesalud.org.ar/](https://avia.cuidadosdesalud.org.ar/)

## 📄 Licencia

Este modelo y su documentación están disponibles para uso en investigaciones y aplicaciones de salud pública. Para más información sobre el uso y licencias, consulte el repositorio principal del proyecto.

## 🔗 Referencias

- Sitio web del proyecto: [https://avia.cuidadosdesalud.org.ar/](https://avia.cuidadosdesalud.org.ar/)
- Documentación técnica: Disponible en el repositorio principal

## 📞 Contacto

Para preguntas, sugerencias o soporte técnico, por favor contacte a través de:
- Sitio web: [https://avia.cuidadosdesalud.org.ar/](https://avia.cuidadosdesalud.org.ar/)
- Email: (disponible en el sitio web)

---

**Nota importante**: Este modelo está diseñado como herramienta de apoyo clínico y no debe reemplazar la evaluación médica profesional. Siempre consulte con profesionales de la salud para diagnósticos y tratamientos.
