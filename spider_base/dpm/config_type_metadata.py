from pymongo import MongoClient
import requests

#基本配置
db_name = "dpm"
col_name = "metadata"

#数据库配置
client = MongoClient()
db = client.get_database(db_name)
col = db.get_collection(col_name)

type_options = col.find({"key":"type_options"})[0]
lists = []
for x in type_options["type_options"]:
    lists.append(x)
print(lists)

newItems = [
    {
        "name": 'bamboos',
        "total_page": 26,
        "keys": ["name","href","img_url","year","type","author"],
        "base_url_num":106
    },
    {
        "name": 'bronzes',
        "total_page": 23,
        "keys": ["name","href","img_url","year","type"],
        "base_url_num":94
    },
    {
        "name": 'ceramics',
        "total_page": 116,
        "keys": ["name","href","img_url","year","type","place"],
        "base_url_num":90
    },
{
        "name": 'clocks',
        "total_page": 23,
        "keys": ["name","href","img_url","year","type","place"],
        "base_url_num":99
    },
{
        "name": 'defenses',
        "total_page": 5,
        "keys": ["name","href","img_url","year"],
        "base_url_num":110
    },
{
        "name": 'embroiders',
        "total_page": 38,
        "keys": ["name","href","img_url","year","type","use"],
        "base_url_num":96
    },
{
        "name": 'enamels',
        "total_page": 13,
        "keys": ["name","href","img_url","year","type"],
        "base_url_num":100
    },
{
        "name": 'foreigns',
        "total_page": 4,
        "keys": ["name","href","img_url","year","place"],
        "base_url_num":114
    },
{
        "name": 'gears',
        "total_page": 17,
        "keys": ["name","href","img_url","year"],
        "base_url_num":98
    },
{
        "name": 'glasses',
        "total_page": 4,
        "keys": ["name","href","img_url","year","type"],
        "base_url_num":105
    },
{
        "name": 'handwritings',
        "total_page": 41,
        "keys": ["name","href","img_url","year","type","author"],
        "base_url_num":92
    },
{
        "name": 'impress',
        "total_page": 26,
        "keys": ["name","href","img_url","year","type"],
        "base_url_num":93
    },
{
        "name": 'jades',
        "total_page": 41,
        "keys": ["name","href","img_url","year","type"],
        "base_url_num":104
    },
{
        "name": 'jewelrys',
        "total_page": 7,
        "keys": ["name","href","img_url","year"],
        "base_url_num":108
    },
{
        "name": 'lacquerwares',
        "total_page": 14,
        "keys": ["name","href","img_url","year","type"],
        "base_url_num":101
    },
{
        "name": 'musics',
        "total_page": 8,
        "keys": ["name","href","img_url","year","type"],
        "base_url_num":112
    },
{
        "name": 'paints',
        "total_page": 107,
        "keys": ["name","href","img_url","year","type","author"],
        "base_url_num":91
    },
{
        "name": 'religions',
        "total_page": 18,
        "keys": ["name","href","img_url","year","type"],
        "base_url_num":107
    },
{
        "name": 'sculptures',
        "total_page": 30,
        "keys": ["name","href","img_url","year","type"],
        "base_url_num":102
    },
{
        "name": 'seals',
        "total_page": 25,
        "keys": ["name","href","img_url","year","author"],
        "base_url_num":95
    },
{
        "name": 'studies',
        "total_page": 15,
        "keys": ["name","href","img_url","year","type"],
        "base_url_num":97
    },
{
        "name": 'tinwares',
        "total_page": 17,
        "keys": ["name","href","img_url","year","type"],
        "base_url_num":103
    },
{
        "name": 'utensils',
        "total_page": 21,
        "keys": ["name","href","img_url","year","type"],
        "base_url_num":113
    },
]

print(newItems)

items = []
for x in newItems:
    for y in lists:
        if y["en_name"] == x["name"]:
            x["cn_name"] = y["name"]
            x["href"] = y["href"]
            x["value"] = y["value"]
            items.append(x)
            break
for x in items:
    print(x)
col.insert_many(items)

