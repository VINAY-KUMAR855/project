import sqlite3
conn=sqlite3.connect("labels.db")
cur=conn.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS label(id INTEGER PRIMARY KEY,label INTEGER UNIQUE,name TEXT UNIQUE)")
label_list=[(0,'Ayrshire cattle'),(1,'Brown Swiss cattle'),(2,'Gir'),(3,'Hallikar Cow'),(4,'Holstein Friesian cattle'),(5,'Jaffrabadi Buffalo'),(6,'Jersey cattle'),(7,'Kankrej Cow'),(8,'Murrah'),(9,'Nagpuri Buffalo'),(10,'Nili ravi Buffalo'),(11,'Rathi Cow'),(12,'Tharparkar Cow'),(13,'sahiwal')]
cur.executemany("INSERT INTO label(label,name) VALUES(?,?)",label_list)
cur.execute("SELECT * FROM label")
rows=cur.fetchall()
for row in rows:
	print(row)
print(len(rows))
conn.commit()
conn.close()

