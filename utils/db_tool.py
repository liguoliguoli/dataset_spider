def get_col(db_name, col_name):
    from pymongo import MongoClient
    client = MongoClient()
    db = client.get_database(db_name)
    col = db.get_collection(col_name)
    return col