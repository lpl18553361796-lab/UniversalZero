class dotdict(dict):
    """
    让字典支持点号访问 (dot access) 的小工具。
    例如：args.num 而不是 args['num']
    """
    def __getattr__(self, name):
        return self[name]
