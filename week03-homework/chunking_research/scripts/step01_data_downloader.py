"""
数据下载器模块 (Data Downloader Module)

该模块提供了一个通用的数据下载器类，用于从多个来源下载研究数据：
- 维基百科文章 (Wikipedia articles)
- arXiv 学术论文 (arXiv academic papers)

主要特性：
- 面向对象设计，易于扩展和维护
- 完整的错误处理和日志记录
- 批量下载功能
- 灵活的配置选项
- 下载状态监控

依赖包安装：
pip install wikipedia-api arxiv

使用示例：
    downloader = DataDownloader()
    results = downloader.download_all_default_data()
"""

# pip install wikipedia-api arxiv
import os
import wikipediaapi
import arxiv
from typing import List, Optional
import logging


class DataDownloader:
    """数据下载器类，用于下载维基百科文章和arXiv论文"""
    
    def __init__(self, 
                 project_root: Optional[str] = None,
                 user_agent: str = 'ChunkingResearch/1.0 (your.email@domain.com)',
                 language: str = 'en'):
        """
        初始化数据下载器
        
        Args:
            project_root: 项目根目录路径，如果为None则自动检测
            user_agent: Wikipedia API的用户代理
            language: Wikipedia的语言代码
        """
        self.user_agent = user_agent
        self.language = language
        
        # 设置项目根目录和数据目录
        if project_root is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.project_root = os.path.dirname(script_dir)
        else:
            self.project_root = project_root
            
        self.data_dir = os.path.join(self.project_root, 'data', 'raw')
        
        # 初始化Wikipedia API
        self.wiki = wikipediaapi.Wikipedia(self.user_agent, self.language)
        
        # 设置日志
        self._setup_logging()
        
        # 确保数据目录存在
        self._ensure_data_directory()
    
    def _setup_logging(self):
        """设置日志配置"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _ensure_data_directory(self):
        """确保数据目录存在"""
        os.makedirs(self.data_dir, exist_ok=True)
        self.logger.info(f"数据目录: {self.data_dir}")
    
    def download_wikipedia_page(self, topic: str) -> bool:
        """
        下载单个维基百科页面
        
        Args:
            topic: 要下载的主题
            
        Returns:
            bool: 下载是否成功
        """
        try:
            page = self.wiki.page(topic)
            if page.exists():
                file_path = os.path.join(self.data_dir, f'{topic}.txt')
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(page.text)
                self.logger.info(f"维基百科页面下载成功: {topic} -> {file_path}")
                return True
            else:
                self.logger.warning(f"维基百科页面不存在: {topic}")
                return False
        except Exception as e:
            self.logger.error(f"下载维基百科页面 '{topic}' 时出错: {e}")
            return False
    
    def download_wikipedia_pages(self, topics: List[str]) -> dict:
        """
        批量下载维基百科页面
        
        Args:
            topics: 要下载的主题列表
            
        Returns:
            dict: 下载结果统计 {'success': int, 'failed': int, 'results': list}
        """
        results = []
        success_count = 0
        failed_count = 0
        
        self.logger.info(f"开始下载 {len(topics)} 个维基百科页面...")
        
        for topic in topics:
            success = self.download_wikipedia_page(topic)
            results.append({'topic': topic, 'success': success})
            if success:
                success_count += 1
            else:
                failed_count += 1
        
        self.logger.info(f"维基百科下载完成: 成功 {success_count}, 失败 {failed_count}")
        return {
            'success': success_count,
            'failed': failed_count,
            'results': results
        }
    
    def download_arxiv_paper(self, query: str, max_results: int = 1) -> bool:
        """
        下载arXiv论文
        
        Args:
            query: 搜索查询
            max_results: 最大结果数
            
        Returns:
            bool: 下载是否成功
        """
        try:
            search = arxiv.Search(query=query, max_results=max_results)
            papers = list(search.results())
            
            if papers:
                paper = papers[0]
                filename = f'{query.replace(" ", "_")}.pdf'
                paper.download_pdf(dirpath=self.data_dir, filename=filename)
                pdf_path = os.path.join(self.data_dir, filename)
                self.logger.info(f"arXiv论文下载成功: {query} -> {pdf_path}")
                return True
            else:
                self.logger.warning(f"未找到arXiv论文: {query}")
                return False
        except Exception as e:
            self.logger.error(f"下载arXiv论文 '{query}' 时出错: {e}")
            return False
    
    def download_arxiv_papers(self, queries: List[str], max_results: int = 1) -> dict:
        """
        批量下载arXiv论文
        
        Args:
            queries: 搜索查询列表
            max_results: 每个查询的最大结果数
            
        Returns:
            dict: 下载结果统计
        """
        results = []
        success_count = 0
        failed_count = 0
        
        self.logger.info(f"开始下载 {len(queries)} 个arXiv论文...")
        
        for query in queries:
            success = self.download_arxiv_paper(query, max_results)
            results.append({'query': query, 'success': success})
            if success:
                success_count += 1
            else:
                failed_count += 1
        
        self.logger.info(f"arXiv下载完成: 成功 {success_count}, 失败 {failed_count}")
        return {
            'success': success_count,
            'failed': failed_count,
            'results': results
        }
    
    def download_all_default_data(self) -> dict:
        """
        下载所有默认数据集
        
        Returns:
            dict: 完整的下载结果统计
        """
        self.logger.info("开始下载默认数据集...")
        
        # 默认的维基百科主题
        wiki_topics = ['Quantum computing', 'Climate change', 'Artificial intelligence']
        
        # 默认的arXiv查询
        arxiv_queries = ['quantum computing', 'climate change', 'AI']
        
        # 下载维基百科页面
        wiki_results = self.download_wikipedia_pages(wiki_topics)
        
        # 下载arXiv论文
        arxiv_results = self.download_arxiv_papers(arxiv_queries)
        
        # 合并结果
        total_results = {
            'wikipedia': wiki_results,
            'arxiv': arxiv_results,
            'total_success': wiki_results['success'] + arxiv_results['success'],
            'total_failed': wiki_results['failed'] + arxiv_results['failed']
        }
        
        self.logger.info(f"数据下载完成! 总计: 成功 {total_results['total_success']}, 失败 {total_results['total_failed']}")
        return total_results
    
    def get_downloaded_files(self) -> List[str]:
        """
        获取已下载文件列表
        
        Returns:
            List[str]: 文件路径列表
        """
        if not os.path.exists(self.data_dir):
            return []
        
        files = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith(('.txt', '.pdf')):
                files.append(os.path.join(self.data_dir, filename))
        
        return sorted(files)
    
    def print_status(self):
        """打印下载器状态信息"""
        files = self.get_downloaded_files()
        print(f"\n=== 数据下载器状态 ===")
        print(f"项目根目录: {self.project_root}")
        print(f"数据目录: {self.data_dir}")
        print(f"已下载文件数: {len(files)}")
        if files:
            print("已下载文件:")
            for file_path in files:
                file_size = os.path.getsize(file_path) / 1024  # KB
                print(f"  - {os.path.basename(file_path)} ({file_size:.1f} KB)")


def main():
    """主函数"""
    # 创建下载器实例
    downloader = DataDownloader()
    
    # 打印初始状态
    downloader.print_status()
    
    # 下载所有默认数据
    results = downloader.download_all_default_data()
    
    # 打印最终状态
    downloader.print_status()
    
    return results


if __name__ == "__main__":
    main()
