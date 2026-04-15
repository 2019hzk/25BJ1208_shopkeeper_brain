TEST_CASES = [

    # ════════════════════════════════════════════════════
    # 一、精确查询（直接确认）
    #    预期：confirmed 有值，options 为空
    # ════════════════════════════════════════════════════

    {
        "query": "RS-12数字万用表怎么测电压？",
        "llm_extract": ["RS-12数字万用表"],
        "expected_confirmed": ["RS-12 数字万用表"],
        "expected_options": [],
        "description": "精确-RS-12测电压"
    },
    {
        "query": "RS-12数字万用表如何测量电阻？",
        "llm_extract": ["RS-12数字万用表"],
        "expected_confirmed": ["RS-12 数字万用表"],
        "expected_options": [],
        "description": "精确-RS-12测电阻"
    },
    {
        "query": "数字万用表RS-12的操作说明",
        "llm_extract": ["数字万用表RS-12"],
        "expected_confirmed": ["RS-12 数字万用表"],
        "expected_options": [],
        "description": "精确-RS-12名称顺序调换"
    },
    {
        "query": "RS12万用表怎么用？",
        "llm_extract": ["RS12万用表"],
        "expected_confirmed": ["RS-12 数字万用表"],
        "expected_options": [],
        "description": "精确-RS12简称（无横杠）"
    },
    {
        "query": "hak180的安全注意事项有哪些？",
        "llm_extract": ["hak180安全注意事项"],
        "expected_confirmed": ["hak180产品安全手册"],
        "expected_options": [],
        "description": "精确-hak180指定安全手册"
    },


    # ════════════════════════════════════════════════════
    # 二、模糊/不够精确查询（给候选让用户确认）
    #    预期：confirmed 为空，options 有值
    # ════════════════════════════════════════════════════

    {
        "query": "RS-12",
        "llm_extract": ["RS-12"],
        "expected_confirmed": [],
        "expected_options": ["RS-12 数字万用表"],
        "description": "模糊-仅型号名"
    },
    {
        "query": "那个RS型号的万用表怎么测电压？",
        "llm_extract": ["RS型号万用表"],
        "expected_confirmed": [],
        "expected_options": ["RS-12 数字万用表"],
        "description": "模糊-RS型号描述"
    },
    {
        "query": "hak开头的万用表安全手册在哪里？",
        "llm_extract": ["hak开头安全手册"],
        "expected_confirmed": [],
        "expected_options": ["hak180产品安全手册"],
        "description": "模糊-hak开头描述"
    },

    # ════════════════════════════════════════════════════
    # 三、同产品多手册（用户未指定，让用户选）
    #    预期：confirmed 为空，options 有值
    # ════════════════════════════════════════════════════

    {
        "query": "hak180万用表怎么使用？",
        "llm_extract": ["hak180万用表"],
        "expected_confirmed": [],
        "expected_options": ["hak180使用说明书", "hak180产品安全手册"],
        "description": "多手册-hak180未指定手册"
    },
    {
        "query": "hak180怎么操作面板和LED指示灯？",
        "llm_extract": ["hak180"],
        "expected_confirmed": [],
        "expected_options": ["hak180使用说明书", "hak180产品安全手册"],
        "description": "多手册-hak180简称未指定"
    },
    {
        "query": "hak180",
        "llm_extract": ["hak180"],
        "expected_confirmed": [],
        "expected_options": ["hak180使用说明书", "hak180产品安全手册"],
        "description": "多手册-仅产品名"
    },

    # ════════════════════════════════════════════════════
    # 四、多商品同时查询（混合场景）
    #    预期：confirmed 有值 + options 有值
    # ════════════════════════════════════════════════════

    {
        "query": "RS-12数字万用表和hak180万用表有什么区别？",
        "llm_extract": ["RS-12数字万用表", "hak180万用表"],
        "expected_confirmed": ["RS-12 数字万用表"],
        "expected_options": ["hak180使用说明书", "hak180产品安全手册"],
        "description": "多商品-RS-12确认+hak180让用户选"
    },

    # ════════════════════════════════════════════════════
    # 五、无法识别（无关查询 + 不存在的商品）
    #    预期：confirmed 和 options 都为空
    # ════════════════════════════════════════════════════

    {
        "query": "你们店里都有什么产品？",
        "llm_extract": [],
        "expected_confirmed": [],
        "expected_options": [],
        "description": "无关-泛化询问"
    },
    {
        "query": "如何选择合适的万用表？",
        "llm_extract": [],
        "expected_confirmed": [],
        "expected_options": [],
        "description": "无关-通用咨询"
    },
    {
        "query": "RS-999万用表怎么使用？",
        "llm_extract": ["RS-999万用表"],
        "expected_confirmed": [],
        "expected_options": [],
        "description": "错误-不存在RS-999"
    },
    {
        "query": "abc123万用表的操作说明",
        "llm_extract": ["abc123万用表"],
        "expected_confirmed": [],
        "expected_options": [],
        "description": "错误-不存在abc123"
    },
]