# Predicción de cancelanción de clientes
Modelo de clasificación para anticipar qué clientes del operador de telecomunicaciones Interconnect tienen mayor probabilidad de cancelar su servicio, con el fin de dirigir campañas de retención (códigos promocionales, planes especiales) de forma proactiva antes de que el cliente se vaya.

📋 Contexto del negocio

Interconnect ofrece servicios de telefonía fija e internet (DSL y fibra óptica), además de servicios adicionales como seguridad en línea, soporte técnico y streaming. El equipo de marketing necesita identificar con anticipación a los clientes en riesgo de cancelación para poder retenerlos, en lugar de reaccionar después de que ya se han ido.

🎯 Objetivo

Construir un modelo de clasificación binaria que prediga la variable Churn (1 = cancela, 0 = permanece activo), maximizando el ROC-AUC como métrica principal —adecuada dado el desbalance de clases (~26.5% de cancelación)— y reportando también Accuracy y F1-score.
