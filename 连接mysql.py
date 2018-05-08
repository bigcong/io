import pymysql

db = pymysql.connect("localhost", "root", "", "12lian")
cursor = db.cursor()
cursor.execute("select * from user  limit 1")
data = cursor.fetchall()

print("Database version : %s " % data[0][0])
