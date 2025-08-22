import sys # para leer argumentos desde la línea de comandos
import mysql.connector # conector mySql
from datetime import datetime # activa fecha y hora para registrarla

if len(sys.argv) > 1: # si me pasan el número de galletas contadas
    cantidad = int(sys.argv[1]) # guarda en cantidad el argumento que viene del script app en self.total
else:
    with open("conteo.txt", "r") as f: # si no, abro el archivo donde se guardo el último conteo
        cantidad = int(f.read().strip()) # se lee el valor y se convierte a entero

conn = mysql.connector.connect( # se abre conexión en la base de datos en local y se da el password de Mysql que corre en mi pc
    host="localhost",
    user="root",
    password="Aa2283287",
    database="inventario"
)

cursor = conn.cursor() # creo el cursor para ejecutar el código SQL y se crea la tabla si no existe
cursor.execute("""
    CREATE TABLE IF NOT EXISTS conteo_galletas ( 
        id INT AUTO_INCREMENT PRIMARY KEY,
        fecha DATETIME,
        cantidad INT
    )
""")
cursor.execute( # inserto el registro cantidad, creado por la app con fecha y hora
    "INSERT INTO conteo_galletas (fecha, cantidad) VALUES (%s, %s)",
    (datetime.now(), cantidad)
)
conn.commit()  # Confirmo y guardo
cursor.close() # Cierro el cursor
conn.close()   # cierro conexión

print(f"✅ Insertado {cantidad} galletas en la BD.") # mensaje de insertado
