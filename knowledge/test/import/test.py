from typing import Dict, List
from collections import defaultdict

_tasks_running_list: Dict[str, List[str]] = defaultdict(list)

value=_tasks_running_list['task_id']

print(value)
