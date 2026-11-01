# AETERNA-V | Protocolo de reproducibilidad

1. Congelar commit, dependencias, plataforma y semillas aleatorias.
2. Ejecutar la batería de pruebas y conservar logs completos y código de salida.
3. Calcular SHA-256 para scripts, entradas y resultados.
4. Reportar tolerancias, precisión numérica y configuración exacta.
5. Comparar contra baselines CPU/GPU y separar consumo del núcleo del sistema completo.
6. Documentar fallos, ruido y límites de aplicación.

**Regla:** aprobar un test solo valida la propiedad cubierta por ese test; no equivale a una certificación física.
