from pathlib import Path
from knowledge.processor.import_processor.base import BaseNode, setup_logging, T
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.processor.import_processor.exceptions import StateFieldError, ValidationError


class EntryNode(BaseNode):
    name = "entry_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        根据上传文件的后缀 修改state中 is_md_read_enabled  is_pdf_read_enabled
        Args:
            state:

        Returns:

        """

        # 1. 获取上文的文件
        import_file_path = state.get('import_file_path', '')
        file_dir = state.get('file_dir', '')

        # 2. 判断
        if not import_file_path:
            raise StateFieldError(node_name=self.name, field_name='import_file_path', expected_type=str)
        if not file_dir:
            raise StateFieldError(node_name=self.name, field_name='file_dir', expected_type=str)

        # 3. Path标准化
        import_file_path_obj = Path(import_file_path)
        file_dir_obj = Path(file_dir)

        # 4. 判读
        if not import_file_path_obj.exists():
            raise StateFieldError(node_name=self.name, field_name='import_file_path_obj', expected_type=Path)
        if not file_dir_obj.exists():
            raise StateFieldError(node_name=self.name, field_name='file_dir_obj', expected_type=Path)

        # 5. 获取文件的后缀
        if import_file_path_obj.suffix == '.pdf':
            state['is_pdf_read_enabled'] = True
        elif import_file_path_obj.suffix == '.md':
            state['is_md_read_enabled'] = True
        else:
            self.logger.error(f"该文件后缀格式{import_file_path_obj.suffix}不支持")
            raise ValidationError(message=f"该文件的后缀格式{import_file_path_obj.suffix}不支持", node_name=self.name)

        # 6. 获取上传文件的标题，更新到state中
        state['file_title'] = import_file_path_obj.stem

        # 7. 返回state
        return state


if __name__ == '__main__':
    entry_node = EntryNode()
    init_state = {
        # "import_file_path": r"D:\develop\develop\workspace\pycharm\BJ251208\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\万用表的使用.pdf",
        "import_file_path": r"D:\develop\develop\workspace\pycharm\BJ251208\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\万用表的使用\hybrid_auto\万用表的使用.md",
        "file_dir": r"D:\develop\develop\workspace\pycharm\BJ251208\shopkeeper_brain\knowledge\processor\import_processor\temp_dir"
    }
    result = entry_node.process(init_state)

    import json

    print(json.dumps(result, ensure_ascii=False, indent=4))
