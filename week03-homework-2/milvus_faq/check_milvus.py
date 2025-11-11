#!/usr/bin/env python3
"""简单脚本：检查 Milvus 连接 & 指定 collection 的实体数量

用法:
  python check_milvus.py            # 使用默认 tcp://127.0.0.1:19530 和 collection 'faq_demo'
  MILVUS_URI='tcp://host:19530' python check_milvus.py
  python check_milvus.py my_collection

确保在虚拟环境安装了 pymilvus:
  pip install pymilvus
"""
import os
import sys
from pymilvus import connections, utility

def parse_uri(uri: str):
    host = "127.0.0.1"
    port = 19530
    try:
        if uri.startswith("tcp://"):
            _, addr = uri.split("tcp://", 1)
        else:
            addr = uri
        if ":" in addr:
            host, port_s = addr.split(":")
            port = int(port_s)
    except Exception:
        pass
    return host, port

def main():
    uri = os.getenv("MILVUS_URI", "tcp://127.0.0.1:19530")
    collection = sys.argv[1] if len(sys.argv) > 1 else os.getenv("MILVUS_COLLECTION", "faq_demo")

    host, port = parse_uri(uri)
    print(f"Connecting to Milvus at {host}:{port} (uri={uri})")
    try:
        connections.connect(host=host, port=port)
    except Exception as e:
        print("Failed to connect to Milvus:", e)
        sys.exit(2)

    try:
        cols = utility.list_collections()
        print("Collections:", cols)
    except Exception as e:
        print("Failed to list collections:", e)
        sys.exit(3)

    if collection in cols:
        try:
            from pymilvus import Collection
            # Try to flush if available (some pymilvus versions expose flush in utility)
            try:
                if hasattr(utility, 'flush'):
                    print(f"Calling utility.flush(['{collection}']) ...")
                    utility.flush([collection])
                else:
                    print("utility.flush not available in this pymilvus version; skipping flush")
            except Exception as e:
                print("Flush failed or not supported:", e)

            c = Collection(collection)
            try:
                print("Loading collection into memory ...")
                c.load()
            except Exception as e:
                print("Load failed (may already be loaded or unsupported):", e)

            # show schema and detailed stats to help debug zero-entity cases
            try:
                schema = c.schema
                print("Schema fields:")
                for f in schema.fields:
                    print(" ", f.name, f.dtype)
            except Exception as e:
                print("Failed to read schema:", e)

            try:
                # Prefer utility.get_collection_stats if available
                if hasattr(utility, 'get_collection_stats'):
                    stats = utility.get_collection_stats(collection)
                    print('Collection stats:', stats)
                else:
                    # Fallback: print num_entities and partition list
                    print('num_entities:', c.num_entities)
                    try:
                        parts = utility.list_partitions(collection)
                        print('partitions:', parts)
                    except Exception:
                        pass
            except Exception as e:
                print('Failed to get collection stats:', e)

            print(f"Collection '{collection}' exists. num_entities = {c.num_entities}")
        except Exception as e:
            print(f"Failed to inspect collection '{collection}':", e)
            sys.exit(4)
    else:
        print(f"Collection '{collection}' not found in Milvus.")

if __name__ == '__main__':
    main()
