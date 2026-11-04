class ServiceError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.status = status


class InputTooLong(ServiceError):
    def __init__(self, max_length: int):
        super().__init__(
            "input_too_long",
            f"完整 NLI 文本对超过 {max_length} tokens；请缩短描述。服务不会静默截断。",
            422,
        )
