import re, time, base64
from collections import deque
from dataclasses import dataclass
from logging import Logger
from typing import List, Dict, Tuple, Set, Optional, Deque
from pathlib import Path

from openai import OpenAI

from knowledge.processor.import_processor.base import BaseNode, setup_logging, T
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.processor.import_processor.exceptions import StateFieldError, FileProcessingError
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients


@dataclass  # 未来直接实例化（不需要重写__init__方法 __repr__方法）
class ImageContext:
    """
    一张图片上下文信息

    """
    head: str  # 上文标题内容
    pre_text: str  # 上文内容
    post_text: str  # 下文内容


@dataclass
class ImageInfo:
    """
    一张图片的完整信息
    图片的名字：作为存储图片摘要的字典容器key
    图片的地址：1.vlm要用【xx.png/xxx.png的内容】 2. minio要用
    图片上下文信息：作为VLM使用
    """
    name: str  # 图片的名字（全名）
    path: str  # 图片地址
    imag_context: ImageContext  # 图片上下文信息


class _MdFileHandler:
    """
    主要职责：
    1. 读取md内容、md_path、图片目录
    2. 备份新的md_content.(方便测试观察)
    """

    def __init__(self, logger: Logger, node_name: str):
        self.logger = logger
        self.node_name = node_name

    def validate_and_read_md(self, state) -> Tuple[str, Path, Path]:
        """
        核心逻辑：
        1. 读取md内容
        2. 读取md的路径
        3. 读取图片目录
        Args:
            state:  上一个节点更新后的state

        Returns:
            Tuple[str,Path,Path]

        """

        # 1. 从state中获取md_path
        md_path = state.get('md_path', '')

        # 2. 非空判断
        if not md_path:
            raise StateFieldError(node_name=self.node_name, field_name='md_path', expected_type=str)

        # 3. Path标准化
        md_path_obj = Path(md_path)

        # 4. 判断路径是否存在
        if not md_path_obj.exists():
            raise StateFieldError(node_name=self.node_name, field_name='md_path', expected_type=Path)

        # 5. 读取md_content
        try:
            with open(md_path_obj, 'r', encoding='utf-8') as f:
                md_content = f.read()
        except IOError as e:
            self.logger.error(f"MD文件:{md_path_obj.name} 打开失败")
            raise FileProcessingError(message="文件打开失败", node_name=self.node_name)

        # 6. 获取图片目录
        img_dir = md_path_obj.parent / "images"

        # 7. 返回
        return md_content, md_path_obj, img_dir

    def backup(self, md_path_obj: Path, new_md_content: str) -> str:
        self.logger.info("【step_5】备份新文件")

        new_file_path = md_path_obj.with_name(
            f"{md_path_obj.stem}_new{md_path_obj.suffix}"
        )
        try:
            with open(new_file_path, "w", encoding="utf-8") as f:
                f.write(new_md_content)
            self.logger.info(f"处理后的文件已备份至: {new_file_path}")
        except IOError as e:
            self.logger.error(f"写入新文件失败 {new_file_path}: {e}")
            raise FileProcessingError(
                f"文件写入失败: {e}", node_name="md_img_node"
            )
        return str(new_file_path)


class _ImageScanner:
    """
    主要职责：
    1. 根据图片目录，得到该目录下有效的图片文件
    2. 去到md文件中定位图片的位置
    3. 获取该图片在md中的上下文内容（给VLM模型提供上下文信息，帮助 模型识别结果更加准确）
    4. 最终组装所有图片的上下文内容（List）

    """

    def __init__(self, logger: Logger):
        self.logger = logger

    def scan_imgs_dir(self, img_dir_obj: Path, md_content: str, image_extensions: Set[str], img_content_length: int) -> \
            List[ImageInfo]:
        """
        核心逻辑：
        1. 扫描指定图片目录下的所有图片文件

        2. 遍历每一个图片文件去MD文件中获取到位置（上下文）
        2.1 上文信息（标题 + 上文内容）
        2.2 下文信息（下文内容）

        3. 将每一个（图片的上下文：ImageContext）放到最终封装每一个图片完整信息(ImageInfo)的容器中

        4. 将容器返回
        Args:
            img_dir_obj: 图片目录
            md_content:  md内容
            image_extensions: 允许的图片后缀格式
            img_content_length: 上下文的长度（各自最大不能超过200）
        Returns:
            List[ImageInfo]

        """
        img_info_list = []
        # 1. 遍历图片目录
        for img_path in img_dir_obj.iterdir():

            # 1.1 过滤掉子目录
            if not img_path.is_file():
                self.logger.error(f"{img_path}不是一个有效的文件")
                continue

            # 1.2  过滤掉不合法的图片文件
            if not img_path.suffix in image_extensions:
                self.logger.error(f"{img_path.suffix}不是允许的图片后缀格式")
                continue

            # 1.3 找该图片的上下文
            ctx = self._find_context(img_path.name, md_content, img_content_length)
            if not ctx:
                self.logger.info(f"MD中未找到该图片{img_path.name}引用")
                continue

            # 1.4 封装ImageInfo对象并且放到容器中
            img_info_list.append(ImageInfo(
                name=img_path.name,
                path=str(img_path),
                imag_context=ctx
            ))

        self.logger.info(f"MD中找到{len(img_info_list)}个有效的图片引用")

        # 2. 最终返回
        return img_info_list

    def _find_context(self, img_name: str, md_content: str, img_content_length: int) -> Optional[ImageContext]:
        """
        查找图片的上下文
        Args:
            img_name: 图片名
            md_content: MD内容
            img_content_length:上下文长度

        Returns:
          找到了--->ImageContext:图片的上下文信息
          没找到--->None
        """

        # 1. 预编译正则规则(主要目的：从MD（很多行）中抓取到当前这个图片)
        # ![](images\xxx.png "abc")
        # 正则在大模型应用中特别多
        # . 任意字符 * 0次或者多次  \[ \] \( \) ?非贪婪模式  escape（a.png）
        pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(img_name) + r".*?\)")

        # 2. 按行切割md_content
        md_lines = md_content.split("\n")

        # 3. 遍历每一行以及对应的行索引
        for md_idx, md_line in enumerate(md_lines):

            # 3.1 当前行不是当前图片
            if not pattern.search(md_line):
                continue

            # 3.2 当前行包含当前图片
            # 上文
            # 上文标题的索引作为起始索引(取不到)
            head, prev_index = self._find_heading_up(md_lines, md_idx)
            pre_lines = md_lines[prev_index + 1:md_idx]
            pre_context = self._extract_limited_context(pre_lines, img_content_length, direction="front")

            # 下文
            # 下文标题的索引作为结束索引
            next_index = self._find_heading_down(md_lines, md_idx)
            next_lines = md_lines[md_idx + 1:next_index]
            post_context = self._extract_limited_context(next_lines, img_content_length, direction="back")

            return ImageContext(
                head=head,
                pre_text=pre_context,
                post_text=post_context
            )
        return None

    def _find_heading_up(self, md_lines: List[str], from_idx: int) -> Tuple[str, int]:
        """

        Args:
            md_lines: 整个MD内容
            from_idx: 图片的索引

        Returns:
          当前图片最近的上文标题内容+索引
        """
        for i in range(from_idx - 1, -1, -1):
            if re.match(r"^#{1,6}\s+", md_lines[i]):
                return md_lines[i], i

        return "", -1

    def _find_heading_down(self, md_lines: List[str], from_idx: int) -> int:
        """

        Args:
            md_lines:  整个MD内容
            from_idx:   图片的索引

        Returns:
              当前图片最近的下文标题索引
        """
        for i in range(from_idx + 1, len(md_lines)):
            if re.match(r"^#{1,6}\s+", md_lines[i]):
                return i

        return len(md_lines)

    def _extract_limited_context(self, extracted_md_lines: List[str], img_content_length: int, direction: str) -> str:
        """
        职责：截取给定的上下文内容
        截取策略：不直接根据字符数暴力截取，采用段落方式截取。最后根据段落的字符数是否达到最大上下文长度选择留取。
        段落的规则：
        ①：自然而然的段落 获取切分后的内容 如果是""空字符串
        ②：人为设计其他图片作为段落（其它图片不要）
        Args:
            extracted_md_lines: 上（下）文
            img_content_length: 上下文长度
            direction: 方向(向上找)

        Returns:
            str:上（下）文的内容
        """
        current_paragraph = []
        paragraphs = []

        # 1. 遍历截取的行
        for line in extracted_md_lines:
            # 1.1 定义自然而然段落的规则
            is_blank_line = not line.strip()

            # 1.2 定义人为设计的图片段落规则
            is_other_image = re.match(
                r"^!\[.*?\]\(.*?\)$", line.strip()
            )

            # 1.3 当前行是空行或者其它图片行
            if is_blank_line or is_other_image:
                if current_paragraph:
                    paragraphs.append("\n".join(current_paragraph))
                    current_paragraph = []
                continue

            # 1.4  当前行不是空行也不是其它图片行
            current_paragraph.append(line)

        # 2. 处理最后的行
        if current_paragraph:
            paragraphs.append("\n".join(current_paragraph))

        # 反转(就近原则)
        if direction == "front":
            paragraphs.reverse()
        # 3. 遍历段落列表(判断长度，已经最终选择留下哪些段落)
        total = 0
        selected = []  # 最终收集到的段落
        for paragraph in paragraphs:
            if total + len(paragraph) > img_content_length and selected:
                break
            selected.append(paragraph)
            total += len(paragraph)

        # 反转（保证收集到的顺序和原文档中顺序一致，方便VLM参考）
        if direction == "front":
            selected.reverse()

        # 4. 将最终段落列表中的段落转成一个字符串
        return "\n\n".join(selected)


class _VLMSummarizer:
    """
    主要职责：
    主要根据每一张图片信息以及每一张图片的上下文信息，生成对应该图片的摘要信息
    """

    def __init__(self, logger: Logger, requests_per_minute: int):
        self.logger = logger
        self.requests_per_minute = requests_per_minute

    def _summary_all(self, document_name: str, img_info_list: List[ImageInfo], vl_model: str) -> Dict[str, str]:
        """
        职责：为所有图片生成摘要
        Args:
            document_name: 文档的名字
            img_info_list: 所有图片信息
            vl_model: vlm模型名字

        Returns:
            Dict[str,str]:{"img_name":"summary"}

        """
        summaries = {}
        request_timestamps: Deque[float] = deque()

        # 1. 获取VLM客户端
        try:
            vlm_client = AIClients.get_vlm_client()
        except Exception as e:
            for img_info in img_info_list:
                summaries[img_info.name] = "暂无摘要"
            return summaries

        # 2.调用VLM 为每一张图片生成摘要
        for img_info in img_info_list:
            # 测试一下
            self._enforce_rate_limit(request_timestamps, self.requests_per_minute)
            summaries[img_info.name] = self._summary_one(document_name, img_info, vlm_client, vl_model)

        self.logger.info(f"生成{len(summaries)}图片摘要")
        return summaries

    def _summary_one(self, document_name: str, img_info: ImageInfo, vlm_client: OpenAI, vl_model: str) -> str:
        """
        调用VLM模型为当前图片生成摘要信息
        Args:
            img_info: 当前图片信息
            vlm_client: vlm客户端
            vl_model: vlm模型名

        Returns:
         str:图片摘要

        """
        # 1. 构造VLM需要的上下文（标题名、上文内容、下文内容）
        parts = [p for p in
                 (img_info.imag_context.head, img_info.imag_context.pre_text, img_info.imag_context.post_text) if p]

        # 2. 构建最终的上下文
        final_context = "\n".join(parts) if parts else "暂无上下文"

        # 3. 根据图片地址获取到图片的内容（二进制字节流）---文本协议认识（base64编码）--->解码（‘utf-8’）--->字符串（文本协议能传输） ---- 根据收到字符串解码（二进制字节流 还原图片内容）
        try:
            with  open(img_info.path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
        except IOError as e:
            self.logger.error(f"读取图片文件{img_info.path} 内容失败: {e}")
            return "暂无图片描述"

        # 4. 利用vlm客户端调用VLM模型
        try:
            resp = vlm_client.chat.completions.create(
                model=vl_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"任务：为Markdown文档中的图片生成一个简短的中文标题。\n"
                                f"背景信息：\n"
                                f"  1. 所属文档标题：\"{document_name}\"\n"
                                f"  2. 图片上下文：{final_context}\n"
                                f"请结合图片内容和上述上下文信息，"
                                f"用中文简要总结这张图片的内容，"
                                f"生成一个精准的中文标题摘要（不要包含图片二字）。"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_data}"
                            },
                        },
                    ],
                }],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"图片摘要生成失败 {img_info.path}: {e}")
            return "暂无图片描述"

    def _enforce_rate_limit(
            self, timestamps: Deque[float],
            max_requests: int,
            window: int = 60,
    ):
        now = time.time()
        while timestamps and now - timestamps[0] >= window:
            timestamps.popleft()

        if len(timestamps) >= max_requests:
            sleep_dur = window - (now - timestamps[0])
            if sleep_dur > 0:
                self.logger.info(
                    f"达到速率限制，暂停 {sleep_dur:.2f} 秒..."
                )
                time.sleep(sleep_dur)
            now = time.time()
            while timestamps and now - timestamps[0] >= window:
                timestamps.popleft()

        timestamps.append(now)


class _ImageUploader:
    """
    主要职责：
    1. 将本地图片上传到MinIO，得到该图片在MinIO中可访问的远程地址
    2. 替换md中的摘要和图片地址
    """

    def __init__(self, logger: Logger):
        self.logger = logger

    def upload_and_replace(self, object_dir_name: str, md_content: str, img_info_list: List[ImageInfo],
                           summaries: Dict[str, str],
                           minio_url: str, minio_bucket_name: str):
        """
        上传文件图片到minio并且更新md中的图片地址以及摘要
        Args:
            object_dir_name:  minio对象目录
            md_content:       md的内容
            img_info_list:    图片信息
            summaries:        图片摘要
            minio_url:        minio地址
            minio_bucket_name: 桶名

        Returns:
            更新后的md内容

        """

        # 1. 上传
        remote_urls = self._upload_all(object_dir_name, img_info_list, minio_url, minio_bucket_name)

        # 2. 更新
        md_content = self._update_md(md_content, summaries, remote_urls)

        return md_content

    def _upload_all(self, object_dir_name: str, img_info_list: List[ImageInfo], minio_url: str,
                    minio_bucket_name: str) -> Dict[str, str]:

        remote_urls = {}
        # 1. 得到MinIO客户端
        try:
            minio_client = StorageClients.get_minio_client()
        except Exception as e:
            for img_info in img_info_list:
                remote_urls[img_info.name] = img_info.path
            return remote_urls

        # 2. 遍历上传每一个
        for img_info in img_info_list:
            object_name = f"{object_dir_name}/{img_info.name}"
            try:
                # 2.1 上传图片到MinIO
                minio_client.fput_object(
                    minio_bucket_name, object_name, img_info.path)
                # 2.2 自己拼装路径
                # http://192.168.200.145:9000/桶名/对象名
                self.logger.info(f"成功将图片{img_info.name}上传到MinIO中")
                remote_urls[img_info.name] = f"{minio_url}/{minio_bucket_name}/{object_name}"
            except Exception as e:
                self.logger.warn(f"上传图片{img_info.name}到MinIO失败，用本地图片地址做兜底")
                remote_urls[img_info.name] = img_info.path

        self.logger.info(f"获取到远程的{len(remote_urls)}图片地址")
        return remote_urls

    def _update_md(self, md_content: str, summaries: Dict[str, str], remote_urls: Dict[str, str]) -> str:
        """
        更新MD中的图片描述和远程图片地址
        Args:
            md_content:  md内容
            summaries:   vlm生成的摘要
            remote_urls: minio生成的url

        Returns:
            新md

        """
        # 利用正则寻找(捕获组：()一个捕获组：group(0) 将整个匹配到的内容放进去 group(1)：图片的摘要 group(2):图片地址)
        pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")

        def replacer(match: re.Match) -> str:
            """

            Args:
                match:

            Returns:
                ![摘要](远程图片地址)
            """

            for img_name, img_summary in summaries.items():
                origin_img_path = match.group(2)
                img_name_in_md = Path(origin_img_path).name
                if img_name == img_name_in_md:
                    return f"![{img_summary}]({remote_urls[img_name]})"
            return match.group(0)

        return pattern.sub(replacer, md_content)


class MarkDownToImgNode(BaseNode):
    """
    主要职责：
    1. 得到四个类的实例对象
    2. 分别调用四个实例对象的处理方法

    """

    def __init__(self):
        super().__init__()  # 显示调用父类的构造方法
        self._md_file_handler = _MdFileHandler(self.logger, self.name)
        self._img_scaner = _ImageScanner(self.logger)
        self._vlm_summarizer = _VLMSummarizer(self.logger, self.config.requests_per_minute)
        self._img_uploader = _ImageUploader(self.logger)

    name = "md_to_img_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        入口逻辑
        :param state:
        :return:
        """
        config = self.config
        # 1. 操作md_file_handler
        self.log_step("step1", "读取MD内容、路径以及图片的目录")
        md_content, md_path_obj, img_dir_obj = self._md_file_handler.validate_and_read_md(state)
        # 1.1 判断图片目录不存在
        if not img_dir_obj.exists():
            state['md_content'] = md_content
            return state

        # 2. 操作_img_scaner
        self.log_step("step2", "准备开始扫描图片目录")
        img_info_list: List[ImageInfo] = self._img_scaner.scan_imgs_dir(img_dir_obj,
                                                                        md_content,
                                                                        config.image_extensions,
                                                                        config.img_content_length)

        # 3. 操作_vlm_summarizer
        self.log_step("step3", "利用VLM提取摘要")
        summaries: Dict[str, str] = self._vlm_summarizer._summary_all(md_path_obj.stem, img_info_list,
                                                                      config.vl_model)

        # 4. 操作_img_uploader
        self.log_step("step4","上传文件到MinIO,且更新MD")
        new_md_content = self._img_uploader.upload_and_replace(md_path_obj.stem, md_content, img_info_list,
                                                               summaries,
                                                               config.get_minio_base_url(),
                                                               config.minio_bucket)

        # 5. 备份调配
        self._md_file_handler.backup(md_path_obj, new_md_content)

        return state


if __name__ == '__main__':
    setup_logging()
    md_img_node = MarkDownToImgNode()
    init_state = {
        "md_path": r"D:\develop\develop\workspace\pycharm\BJ251208\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\万用表的使用\hybrid_auto\万用表的使用.md"
    }
    md_img_node.process(init_state)
