from pymongo import MongoClient

# 连接 MongoDB
client = MongoClient("mongodb://admin:123456@192.168.200.145:27017")
# # 选择数据库（不存在则自动创建）
db = client["mydb"]
# # 选择集合（不存在则自动创建）
collection = db["students"]

print("连接成功！")




# 插入单条
# result = collection.insert_one({
#     "name": "张三",
#     "age": 20,
#     "major": "计算机科学"
# })
# # print(type(result))
# print(f"插入成功，ID: {result.inserted_id}")
#
# # 插入多条
# results = collection.insert_many([
#     {"name": "李四", "age": 22, "major": "软件工程"},
#     {"name": "王五", "age": 21, "major": "计算机科学"},
# ])
# print(f"插入 {len(results.inserted_ids)} 条记录")



# 查询全部
# for doc in collection.find():
#     pass
#     # print(doc)
#
# # 条件查询
# for doc in collection.find({"major": "计算机科学"}):
#     print(doc["name"], doc["age"])
#
# # 查询单条
# # student = collection.find_one({"name": "张三"})
# # print(student)
#
# # 带排序和限制
# for doc in collection.find().sort("age", -1).limit(2):
#     print(doc["name"], doc["age"])



# 更新单条
# result = collection.update_one(
#     {"name": "张三"},           # 查询条件
#     {"$set": {"age": 21}}       # 更新操作
# )
# print(f"匹配 {result.matched_count} 条，修改 {result.modified_count} 条")

# 更新多条
# result = collection.update_many(
#     {"major": "计算机科学"},
#     {"$set": {"status": "在读"}}
# )
# print(f"修改 {result.modified_count} 条")



# 删除单条
# result = collection.delete_one({"name": "王五"})
# print(f"删除 {result.deleted_count} 条")

# # 删除多条
result = collection.delete_many({"age": {"$lt": 23}})
print(f"删除 {result.deleted_count} 条")