import pyodbc
from config.settings import DB_SERVER, DB_NAME


def get_connection():
    conn_str = f"""
    DRIVER={{ODBC Driver 17 for SQL Server}};
    SERVER={DB_SERVER};
    DATABASE={DB_NAME};
    Trusted_Connection=yes;
    """

    conn = pyodbc.connect(conn_str)

    return conn