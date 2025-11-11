"""
文件说明：
本文件实现了一个跨学科知识图谱构建器，使用 Neo4j 数据库存储和管理知识图谱。
主要功能包括：
1. 创建学科领域节点和知识点节点。
2. 建立学科内的归属关系（BELONGS_TO）。
3. 建立跨学科的关联关系（CROSS_DISCIPLINE_LINK）。
4. 提供查询接口以获取知识点及其关联信息。
5. 支持查询学科间的跨学科连接路径。

使用方法：
1. 配置 Neo4j 数据库连接信息（URI、用户名、密码）。
2. 调用 `create_knowledge_graph` 方法创建完整的知识图谱。
3. 使用查询方法获取知识图谱中的信息。

注意：
- 请确保 Neo4j 数据库已启动并可用。
- 如果需要清空数据库，请谨慎使用 `clear_all` 方法。
"""

"""
Neo4j 图谱构建器 - 跨学科知识图谱
"""

from neo4j import GraphDatabase
from typing import Optional


class GraphBuilder:
    """知识图谱构建器"""
    
    def __init__(self, uri: str, user: str, password: str):
        """
        初始化 Neo4j 连接
        
        Args:
            uri: Neo4j 数据库 URI (例如: bolt://localhost:7687)
            user: 数据库用户名
            password: 数据库密码
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        """关闭数据库连接"""
        self.driver.close()
    
    def clear_all(self):
        """清空数据库中的所有数据（谨慎使用）"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("数据库已清空")
    
    def create_knowledge_graph(self):
        """创建完整的跨学科知识图谱"""
        with self.driver.session() as session:
            # 创建学科领域节点
            self._create_fields(session)
            
            # 创建各学科知识点节点
            self._create_math_knowledge_points(session)
            self._create_physics_knowledge_points(session)
            self._create_chemistry_knowledge_points(session)
            self._create_geography_knowledge_points(session)
            self._create_cs_knowledge_points(session)
            
            # 建立学科内关系
            self._create_belongs_to_relationships(session)
            
            # 建立跨学科关联关系
            self._create_cross_discipline_links(session)
            
            print("知识图谱创建完成！")
    
    def _create_fields(self, session):
        """创建各学科领域节点"""
        query = """
        CREATE (math:Field {name: '数学'}),
               (physics:Field {name: '物理'}),
               (chemistry:Field {name: '化学'}),
               (geography:Field {name: '地理'}),
               (cs:Field {name: '信息技术'})
        """
        session.run(query)
        print("✓ 学科领域节点创建完成")
    
    def _create_math_knowledge_points(self, session):
        """创建数学知识点节点"""
        query = """
        CREATE 
        (proportional_function:KnowledgePoint:Math {
            name: '正比例函数', 
            description: 'y=kx形式的函数'
        }),
        (quadratic_function:KnowledgePoint:Math {
            name: '二次函数', 
            description: 'y=ax²+bx+c形式的函数'
        }),
        (exponential_function:KnowledgePoint:Math {
            name: '指数函数', 
            description: 'y=a^x形式的函数'
        }),
        (coordinate_system:KnowledgePoint:Math {
            name: '坐标系', 
            description: '平面直角坐标系'
        }),
        (probability:KnowledgePoint:Math {
            name: '概率', 
            description: '事件发生的可能性'
        }),
        (statistics:KnowledgePoint:Math {
            name: '统计', 
            description: '数据收集、整理、分析'
        })
        """
        session.run(query)
        print("✓ 数学知识点节点创建完成")
    
    def _create_physics_knowledge_points(self, session):
        """创建物理知识点节点"""
        query = """
        CREATE 
        (ohm_law:KnowledgePoint:Physics {
            name: '欧姆定律', 
            description: 'I=U/R'
        }),
        (kinetic_energy:KnowledgePoint:Physics {
            name: '动能', 
            description: '物体运动具有的能量 E_k=1/2mv²'
        }),
        (radioactive_decay:KnowledgePoint:Physics {
            name: '放射性衰变', 
            description: '原子核自发衰变规律'
        }),
        (wave_motion:KnowledgePoint:Physics {
            name: '机械波', 
            description: '波在介质中的传播'
        })
        """
        session.run(query)
        print("✓ 物理知识点节点创建完成")
    
    def _create_chemistry_knowledge_points(self, session):
        """创建化学知识点节点"""
        query = """
        CREATE 
        (reaction_rate:KnowledgePoint:Chemistry {
            name: '化学反应速率', 
            description: '化学反应进行的快慢'
        }),
        (half_life:KnowledgePoint:Chemistry {
            name: '半衰期', 
            description: '放射性元素衰变一半所需时间'
        }),
        (ph_value:KnowledgePoint:Chemistry {
            name: 'pH值', 
            description: '溶液酸碱度指标'
        }),
        (periodic_table:KnowledgePoint:Chemistry {
            name: '元素周期表', 
            description: '化学元素的周期性排列'
        })
        """
        session.run(query)
        print("✓ 化学知识点节点创建完成")
    
    def _create_geography_knowledge_points(self, session):
        """创建地理知识点节点"""
        query = """
        CREATE 
        (earth_coordinate:KnowledgePoint:Geography {
            name: '地球坐标系', 
            description: '经纬度坐标系统'
        }),
        (earthquake_wave:KnowledgePoint:Geography {
            name: '地震波', 
            description: '地震产生的波动'
        }),
        (population_growth:KnowledgePoint:Geography {
            name: '人口增长模型', 
            description: '人口数量变化规律'
        }),
        (climate_data:KnowledgePoint:Geography {
            name: '气候数据分析', 
            description: '气温降水等数据分析'
        })
        """
        session.run(query)
        print("✓ 地理知识点节点创建完成")
    
    def _create_cs_knowledge_points(self, session):
        """创建信息技术知识点节点"""
        query = """
        CREATE 
        (data_visualization:KnowledgePoint:CS {
            name: '数据可视化', 
            description: '用图形展示数据'
        }),
        (coordinate_programming:KnowledgePoint:CS {
            name: '坐标绘图编程', 
            description: '用代码绘制函数图像'
        }),
        (random_simulation:KnowledgePoint:CS {
            name: '随机模拟', 
            description: '用计算机模拟随机事件'
        }),
        (exponential_algorithm:KnowledgePoint:CS {
            name: '指数级算法', 
            description: '时间复杂度为O(2^n)的算法'
        })
        """
        session.run(query)
        print("✓ 信息技术知识点节点创建完成")
    
    def _create_belongs_to_relationships(self, session):
        """建立学科内的 BELONGS_TO 关系"""
        queries = [
            # 数学知识点归属
            """
            MATCH (kp:KnowledgePoint:Math {name: '正比例函数'}), (f:Field {name: '数学'})
            CREATE (kp)-[:BELONGS_TO]->(f)
            """,
            """
            MATCH (kp:KnowledgePoint:Math {name: '二次函数'}), (f:Field {name: '数学'})
            CREATE (kp)-[:BELONGS_TO]->(f)
            """,
            """
            MATCH (kp:KnowledgePoint:Math {name: '指数函数'}), (f:Field {name: '数学'})
            CREATE (kp)-[:BELONGS_TO]->(f)
            """,
            """
            MATCH (kp:KnowledgePoint:Math {name: '坐标系'}), (f:Field {name: '数学'})
            CREATE (kp)-[:BELONGS_TO]->(f)
            """,
            """
            MATCH (kp:KnowledgePoint:Math {name: '概率'}), (f:Field {name: '数学'})
            CREATE (kp)-[:BELONGS_TO]->(f)
            """,
            """
            MATCH (kp:KnowledgePoint:Math {name: '统计'}), (f:Field {name: '数学'})
            CREATE (kp)-[:BELONGS_TO]->(f)
            """,
            
            # 物理知识点归属
            """
            MATCH (kp:KnowledgePoint:Physics {name: '欧姆定律'}), (f:Field {name: '物理'})
            CREATE (kp)-[:BELONGS_TO]->(f)
            """,
            """
            MATCH (kp:KnowledgePoint:Physics {name: '动能'}), (f:Field {name: '物理'})
            CREATE (kp)-[:BELONGS_TO]->(f)
            """,
            """
            MATCH (kp:KnowledgePoint:Physics {name: '放射性衰变'}), (f:Field {name: '物理'})
            CREATE (kp)-[:BELONGS_TO]->(f)
            """,
            """
            MATCH (kp:KnowledgePoint:Physics {name: '机械波'}), (f:Field {name: '物理'})
            CREATE (kp)-[:BELONGS_TO]->(f)
            """,
            
            # 化学知识点归属
            """
            MATCH (kp:KnowledgePoint:Chemistry {name: '化学反应速率'}), (f:Field {name: '化学'})
            CREATE (kp)-[:BELONGS_TO]->(f)
            """,
            """
            MATCH (kp:KnowledgePoint:Chemistry {name: '半衰期'}), (f:Field {name: '化学'})
            CREATE (kp)-[:BELONGS_TO]->(f)
            """,
            """
            MATCH (kp:KnowledgePoint:Chemistry {name: 'pH值'}), (f:Field {name: '化学'})
            CREATE (kp)-[:BELONGS_TO]->(f)
            """,
            """
            MATCH (kp:KnowledgePoint:Chemistry {name: '元素周期表'}), (f:Field {name: '化学'})
            CREATE (kp)-[:BELONGS_TO]->(f)
            """,
            
            # 地理知识点归属
            """
            MATCH (kp:KnowledgePoint:Geography {name: '地球坐标系'}), (f:Field {name: '地理'})
            CREATE (kp)-[:BELONGS_TO]->(f)
            """,
            """
            MATCH (kp:KnowledgePoint:Geography {name: '地震波'}), (f:Field {name: '地理'})
            CREATE (kp)-[:BELONGS_TO]->(f)
            """,
            """
            MATCH (kp:KnowledgePoint:Geography {name: '人口增长模型'}), (f:Field {name: '地理'})
            CREATE (kp)-[:BELONGS_TO]->(f)
            """,
            """
            MATCH (kp:KnowledgePoint:Geography {name: '气候数据分析'}), (f:Field {name: '地理'})
            CREATE (kp)-[:BELONGS_TO]->(f)
            """,
            
            # 信息技术知识点归属
            """
            MATCH (kp:KnowledgePoint:CS {name: '数据可视化'}), (f:Field {name: '信息技术'})
            CREATE (kp)-[:BELONGS_TO]->(f)
            """,
            """
            MATCH (kp:KnowledgePoint:CS {name: '坐标绘图编程'}), (f:Field {name: '信息技术'})
            CREATE (kp)-[:BELONGS_TO]->(f)
            """,
            """
            MATCH (kp:KnowledgePoint:CS {name: '随机模拟'}), (f:Field {name: '信息技术'})
            CREATE (kp)-[:BELONGS_TO]->(f)
            """,
            """
            MATCH (kp:KnowledgePoint:CS {name: '指数级算法'}), (f:Field {name: '信息技术'})
            CREATE (kp)-[:BELONGS_TO]->(f)
            """
        ]
        
        for query in queries:
            session.run(query)
        
        print("✓ 学科内关系创建完成")
    
    def _create_cross_discipline_links(self, session):
        """建立跨学科关联关系"""
        queries = [
            # 数学 → 物理
            """
            MATCH (from:KnowledgePoint:Physics {name: '欧姆定律'}),
                  (to:KnowledgePoint:Math {name: '正比例函数'})
            CREATE (from)-[:CROSS_DISCIPLINE_LINK {type: '数学模型应用'}]->(to)
            """,
            """
            MATCH (from:KnowledgePoint:Physics {name: '动能'}),
                  (to:KnowledgePoint:Math {name: '二次函数'})
            CREATE (from)-[:CROSS_DISCIPLINE_LINK {type: '函数关系'}]->(to)
            """,
            """
            MATCH (from:KnowledgePoint:Physics {name: '放射性衰变'}),
                  (to:KnowledgePoint:Math {name: '指数函数'})
            CREATE (from)-[:CROSS_DISCIPLINE_LINK {type: '数学模型'}]->(to)
            """,
            
            # 数学 → 化学
            """
            MATCH (from:KnowledgePoint:Chemistry {name: '半衰期'}),
                  (to:KnowledgePoint:Math {name: '指数函数'})
            CREATE (from)-[:CROSS_DISCIPLINE_LINK {type: '数学模型'}]->(to)
            """,
            """
            MATCH (from:KnowledgePoint:Chemistry {name: '化学反应速率'}),
                  (to:KnowledgePoint:Math {name: '统计'})
            CREATE (from)-[:CROSS_DISCIPLINE_LINK {type: '数据分析'}]->(to)
            """,
            """
            MATCH (from:KnowledgePoint:Chemistry {name: 'pH值'}),
                  (to:KnowledgePoint:Math {name: '指数函数'})
            CREATE (from)-[:CROSS_DISCIPLINE_LINK {type: '对数函数'}]->(to)
            """,
            
            # 数学 → 地理
            """
            MATCH (from:KnowledgePoint:Geography {name: '地球坐标系'}),
                  (to:KnowledgePoint:Math {name: '坐标系'})
            CREATE (from)-[:CROSS_DISCIPLINE_LINK {type: '坐标系统'}]->(to)
            """,
            """
            MATCH (from:KnowledgePoint:Geography {name: '地震波'}),
                  (to:KnowledgePoint:Physics {name: '机械波'})
            CREATE (from)-[:CROSS_DISCIPLINE_LINK {type: '波动方程'}]->(to)
            """,
            """
            MATCH (from:KnowledgePoint:Geography {name: '人口增长模型'}),
                  (to:KnowledgePoint:Math {name: '指数函数'})
            CREATE (from)-[:CROSS_DISCIPLINE_LINK {type: '指数模型'}]->(to)
            """,
            """
            MATCH (from:KnowledgePoint:Geography {name: '气候数据分析'}),
                  (to:KnowledgePoint:Math {name: '统计'})
            CREATE (from)-[:CROSS_DISCIPLINE_LINK {type: '统计分析'}]->(to)
            """,
            
            # 数学 → 信息技术
            """
            MATCH (from:KnowledgePoint:CS {name: '数据可视化'}),
                  (to:KnowledgePoint:Math {name: '坐标系'})
            CREATE (from)-[:CROSS_DISCIPLINE_LINK {type: '图形表达'}]->(to)
            """,
            """
            MATCH (from:KnowledgePoint:CS {name: '坐标绘图编程'}),
                  (to:KnowledgePoint:Math {name: '坐标系'})
            CREATE (from)-[:CROSS_DISCIPLINE_LINK {type: '算法实现'}]->(to)
            """,
            """
            MATCH (from:KnowledgePoint:CS {name: '随机模拟'}),
                  (to:KnowledgePoint:Math {name: '概率'})
            CREATE (from)-[:CROSS_DISCIPLINE_LINK {type: '概率实现'}]->(to)
            """,
            """
            MATCH (from:KnowledgePoint:CS {name: '指数级算法'}),
                  (to:KnowledgePoint:Math {name: '指数函数'})
            CREATE (from)-[:CROSS_DISCIPLINE_LINK {type: '复杂度分析'}]->(to)
            """
        ]
        
        for query in queries:
            session.run(query)
        
        print("✓ 跨学科关联关系创建完成")
    
    def query_knowledge_point(self, name: str):
        """
        查询指定知识点及其关联
        
        Args:
            name: 知识点名称
        """
        with self.driver.session() as session:
            query = """
            MATCH (kp:KnowledgePoint {name: $name})
            OPTIONAL MATCH (kp)-[r:CROSS_DISCIPLINE_LINK]->(related)
            OPTIONAL MATCH (kp)-[:BELONGS_TO]->(field:Field)
            RETURN kp, field, collect({relation: type(r), related: related, type: r.type}) as links
            """
            result = session.run(query, name=name)
            return result.single()
    
    def get_cross_discipline_paths(self, from_field: str, to_field: str):
        """
        查询两个学科之间的跨学科连接路径
        
        Args:
            from_field: 起始学科
            to_field: 目标学科
        """
        with self.driver.session() as session:
            query = """
            MATCH path = (kp1:KnowledgePoint)-[:BELONGS_TO]->(f1:Field {name: $from_field}),
                         (kp2:KnowledgePoint)-[:BELONGS_TO]->(f2:Field {name: $to_field}),
                         (kp1)-[r:CROSS_DISCIPLINE_LINK]->(kp2)
            RETURN kp1.name as from_knowledge, kp2.name as to_knowledge, r.type as link_type
            """
            result = session.run(query, from_field=from_field, to_field=to_field)
            return [record for record in result]


def main():
    """主函数 - 演示如何使用 GraphBuilder"""
    
    # 配置 Neo4j 连接信息
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "test1234"  # 请修改为你的密码
    
    # 创建图谱构建器
    builder = GraphBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        # 清空现有数据（可选，谨慎使用）
        # builder.clear_all()
        
        # 创建知识图谱
        builder.create_knowledge_graph()
        
        print("\n" + "="*50)
        print("示例查询：")
        print("="*50)
        
        # 查询示例：查看欧姆定律的跨学科关联
        result = builder.query_knowledge_point("欧姆定律")
        if result:
            print(f"\n知识点: {result['kp']['name']}")
            print(f"所属学科: {result['field']['name']}")
            print("跨学科关联:")
            for link in result['links']:
                if link['related']:
                    print(f"  → {link['related']['name']} ({link['type']})")
        
        # 查询示例：查看物理与数学之间的跨学科连接
        print("\n物理 → 数学 的跨学科连接:")
        paths = builder.get_cross_discipline_paths("物理", "数学")
        for path in paths:
            print(f"  {path['from_knowledge']} --[{path['link_type']}]--> {path['to_knowledge']}")
        
    finally:
        # 关闭连接
        builder.close()


if __name__ == "__main__":
    main()
