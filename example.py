"""
Ejemplo de uso del modelo AVIA para predicción de fragilidad en adultos mayores.

Este script demuestra cómo cargar el modelo y realizar predicciones
sobre el riesgo de fragilidad basado en datos de salud de un paciente.
"""

from joblib import load
import pandas as pd
import os


def load_model(model_path='model_pipeline.pkl'):
    """
    Carga el modelo entrenado desde un archivo pickle.
    
    Args:
        model_path (str): Ruta al archivo del modelo. Default: 'model_pipeline.pkl'
    
    Returns:
        Pipeline: Modelo entrenado de XGBoost
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"El archivo del modelo no se encontró en: {model_path}\n"
            "Por favor, asegúrese de que el archivo model_pipeline.pkl esté en el mismo directorio."
        )
    
    print(f"Cargando modelo desde: {model_path}")
    pipeline = load(model_path)
    print("✅ Modelo cargado exitosamente")
    return pipeline


def create_example_data():
    """
    Crea un ejemplo de datos de paciente para demostración.
    
    Returns:
        dict: Diccionario con los datos del paciente en español
    """
    return {
        # Demográficos
        'edad': 72,
        'estado_civil': 'casado',  # 'soltero', 'casado', 'concubino', 'divorciado', 'viudo'
        'sexo': 'varon',  # 'varon', 'mujer'
        'escolaridad': '4',  # '1' (universidad), '2' (técnico), '3' (bachillerato), 
                            # '4' (secundaria), '5' (primaria), '6' (otro), '7' (sin estudios)
        'ingresos_brutos': 15000.0,
        
        # Condiciones de salud (0 = no, 1 = sí)
        'CV_HTA': 1.0,  # Hipertensión
        'CV_stroke': 0.0,  # ACV
        'CV_angina': 0.0,  # Angina
        'CV_ICC': 0.0,  # Insuficiencia cardíaca
        'diabetes': 1.0,
        'EPOC': 0.0,
        'artrosis': 1.0,
        'osteoporosis': 0.0,
        'in_urinaria': 2,  # Incontinencia urinaria
        'd_mentales': 0.0,  # Desórdenes mentales
        
        # Hábitos
        'fuma': None,  # Puede ser None (no fumador) o un valor numérico
        'alcohol': 3.0,  # Consumo de alcohol
        
        # Medidas físicas
        'obesidad_abdominal': 105.5,  # Circunferencia abdominal en cm
        'indice_masa_corporal': 28.5,
        'altura': 175.0,  # en cm
        'peso': 87.3,  # en kg
        
        # Funcionalidad sensorial (1 = buena, 2 = regular, 3 = mala)
        'audicion': 2,
        'vision': 2,
        
        # Funcionalidad física
        'caidas': 1.0,  # Ha tenido caídas (0/1)
        'equilibrio': 1.0,  # Problemas de equilibrio (puede ser None)
        'test_silla': 8.5,  # Tiempo en segundos para levantarse de la silla
        
        # Estado de salud percibido
        'estado_salud': 3.0,  # Escala numérica
        'dolor': 2.0,  # Nivel de dolor
        'memoria': 4.0,  # Memoria percibida
        'suenio': 2.0,  # Calidad del sueño
        'soledad': 2.0,  # Nivel de soledad
        
        # Uso de tecnología (0 = no, 1 = sí)
        'usa_internet_email': 0.0,
        'tiene_celular': 1.0,
        
        # Factores sociales
        'soporte_social': 2.0,  # Nivel de soporte social (puede ser None)
        'demencia': 0.0,
        'depresion': 1.0,  # Puede ser None
        
        # Actividad física
        'actividad_fisica_1': 1.0,  # Vigorosa
        'actividad_fisica_2': 1.0,  # Moderada
        'actividad_fisica_3': 2.0,  # Ligera
        
        # Fatigabilidad
        'fatigabilidad_1': 2.0,
        'fatigabilidad_2': 2.0,
        
        # Medidas físicas adicionales
        'fuerza_mano_d_promedio': 35.5,  # Fuerza de mano promedio
        'tiempo_caminar_promedio': 3.2,  # Tiempo promedio de caminata
    }


def predict_frailty(pipeline, data):
    """
    Realiza una predicción de fragilidad usando el modelo.
    
    Args:
        pipeline: Modelo entrenado de XGBoost
        data (dict): Diccionario con los datos del paciente
    
    Returns:
        dict: Diccionario con los resultados de la predicción
    """
    # Convertir datos a DataFrame
    df = pd.DataFrame([data])
    
    # Realizar predicción
    probability = pipeline.predict_proba(df)[:, 1][0]
    risk_score = float(probability)
    
    # Interpretar resultado
    if risk_score < 0.33:
        risk_level = 'bajo'
        diagnosis = 'robusto'
        interpretation = 'El paciente presenta un bajo riesgo de fragilidad'
    elif risk_score < 0.66:
        risk_level = 'medio'
        diagnosis = 'pre-frágil'
        interpretation = 'El paciente presenta un riesgo moderado de fragilidad'
    else:
        risk_level = 'alto'
        diagnosis = 'frágil'
        interpretation = 'El paciente presenta un alto riesgo de fragilidad'
    
    return {
        'risk_score': risk_score,
        'risk_level': risk_level,
        'diagnosis': diagnosis,
        'interpretation': interpretation,
        'probability': probability
    }


def print_results(results, data):
    """
    Imprime los resultados de la predicción de forma legible.
    
    Args:
        results (dict): Resultados de la predicción
        data (dict): Datos del paciente utilizados
    """
    print("\n" + "="*60)
    print("RESULTADOS DE LA PREDICCIÓN DE FRAGILIDAD")
    print("="*60)
    
    print(f"\n📊 Probabilidad de fragilidad: {results['probability']:.4f} ({results['probability']*100:.2f}%)")
    print(f"🎯 Nivel de riesgo: {results['risk_level'].upper()}")
    print(f"🏥 Diagnóstico: {results['diagnosis'].upper()}")
    print(f"\n💡 Interpretación: {results['interpretation']}")
    
    # Mostrar algunos datos relevantes del paciente
    print("\n" + "-"*60)
    print("DATOS DEL PACIENTE (Resumen)")
    print("-"*60)
    print(f"Edad: {data['edad']} años")
    print(f"Sexo: {data['sexo']}")
    print(f"Estado civil: {data['estado_civil']}")
    print(f"IMC: {data.get('indice_masa_corporal', 'N/A')}")
    print(f"Circunferencia abdominal: {data.get('obesidad_abdominal', 'N/A')} cm")
    
    # Condiciones de salud
    conditions = []
    if data.get('CV_HTA') == 1.0:
        conditions.append("Hipertensión")
    if data.get('diabetes') == 1.0:
        conditions.append("Diabetes")
    if data.get('artrosis') == 1.0:
        conditions.append("Artrosis")
    if conditions:
        print(f"Condiciones de salud: {', '.join(conditions)}")
    
    print("="*60 + "\n")


def main():
    """
    Función principal que ejecuta el ejemplo completo.
    """
    print("="*60)
    print("MODELO AVIA - PREDICCIÓN DE FRAGILIDAD")
    print("Ejemplo de uso del modelo")
    print("="*60)
    
    try:
        # Cargar modelo
        pipeline = load_model('model_pipeline.pkl')
        
        # Crear datos de ejemplo
        print("\n📋 Creando datos de ejemplo...")
        data = create_example_data()
        
        # Realizar predicción
        print("🔮 Realizando predicción...")
        results = predict_frailty(pipeline, data)
        
        # Mostrar resultados
        print_results(results, data)
        
        # Ejemplo con múltiples pacientes
        print("\n" + "="*60)
        print("EJEMPLO CON MÚLTIPLES PACIENTES")
        print("="*60)
        
        patients = [
            {'nombre': 'Paciente 1 (Ejemplo actual)', 'data': data},
            {'nombre': 'Paciente 2 (Alto riesgo)', 'data': create_high_risk_patient()},
            {'nombre': 'Paciente 3 (Bajo riesgo)', 'data': create_low_risk_patient()},
        ]
        
        for patient in patients:
            results = predict_frailty(pipeline, patient['data'])
            print(f"\n{patient['nombre']}:")
            print(f"  - Probabilidad: {results['probability']:.4f}")
            print(f"  - Diagnóstico: {results['diagnosis']}")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Sugerencia: Asegúrese de tener el archivo model_pipeline.pkl en el mismo directorio que este script.")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()


def create_high_risk_patient():
    """Crea datos de ejemplo para un paciente de alto riesgo."""
    data = create_example_data()
    data.update({
        'edad': 85,
        'CV_HTA': 1.0,
        'diabetes': 1.0,
        'artrosis': 1.0,
        'caidas': 1.0,
        'estado_salud': 4.0,
        'test_silla': 15.0,
        'actividad_fisica_1': 0.0,
        'actividad_fisica_2': 0.0,
        'actividad_fisica_3': 1.0,
    })
    return data


def create_low_risk_patient():
    """Crea datos de ejemplo para un paciente de bajo riesgo."""
    data = create_example_data()
    data.update({
        'edad': 65,
        'CV_HTA': 0.0,
        'diabetes': 0.0,
        'artrosis': 0.0,
        'caidas': 0.0,
        'estado_salud': 1.0,
        'test_silla': 5.0,
        'actividad_fisica_1': 3.0,
        'actividad_fisica_2': 3.0,
        'actividad_fisica_3': 3.0,
        'obesidad_abdominal': 90.0,
        'indice_masa_corporal': 23.0,
    })
    return data


if __name__ == '__main__':
    main()
