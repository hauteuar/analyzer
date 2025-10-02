import pyodbc

# Test with your driver
conn_str = (
    "DRIVER={IBM DB2 ODBC DRIVER - DB2COPY1};"
    "DATABASE=your_db;"
    "HOSTNAME=your_host;"
    "PORT=50000;"
    "PROTOCOL=TCPIP;"
    "UID=your_user;"
    "PWD=your_password;"
)

try:
    conn = pyodbc.connect(conn_str)
    print("✅ Connected!")
    conn.close()
except Exception as e:
    print(f"❌ Error: {e}")