import random
import string


class AdvancedNameObfuscator:
    def __init__(
        self,
        mode: str = "chaos",
        lang: str = "python",
        rng: random.Random | None = None,
    ) -> None:
        self.mode = mode
        self.lang = lang.lower()
        self.rng = rng if rng is not None else random.Random()
        self.used_names: set[str] = set()
        self._strategy_pool: list[str] = []
        self.injection_words = [
            "IGNORE_CONTEXT",
            "END_OF_TEXT",
            "SYSTEM_ERROR",
            "None_Value",
            "Delete_All",
            "Return_False",
            "Stop_Generation",
            "User_Input",
            "DO_NOT_EXECUTE",
            "void_ptr",
            "null_pointer_exception",
            "unreachable_code",
            "syntax_error",
            "undefined_behavior",
        ]

    def _pick_strategy(self) -> str:
        if self.mode != "chaos":
            return self.mode

        available = ["token_bomb", "injection", "looks_like"]
        if self.lang == "python":
            available.append("underscore")

        if not self._strategy_pool:
            self.rng.shuffle(available)
            self._strategy_pool = available[:]

        return self._strategy_pool.pop()

    def get_name(self) -> str:
        strategy = self._pick_strategy()
        while True:
            if strategy == "token_bomb":
                name = self._token_bomb()
            elif strategy == "injection":
                name = self._injection()
            elif strategy == "underscore":
                name = self._underscore()
            else:
                name = self._looks_like()

            if name not in self.used_names:
                self.used_names.add(name)
                return name

    def _token_bomb(self) -> str:
        length = self.rng.randint(15, 20)
        prefix = self.rng.choice(string.ascii_letters)
        chars = string.ascii_letters + string.digits + "_"
        body = "".join(self.rng.choice(chars) for _ in range(length))
        return prefix + body

    def _injection(self) -> str:
        base = self.rng.choice(self.injection_words)
        suffix = str(self.rng.randint(10, 99))
        return f"{base}_{suffix}"

    def _underscore(self) -> str:
        if self.lang != "python":
            return self._looks_like()

        length = self.rng.randint(1, 10)
        if self.rng.random() < 0.9:
            return "_" * length
        return (
            "_" * self.rng.randint(1, 3)
            + str(self.rng.randint(0, 9))
            + "_" * self.rng.randint(1, 3)
        )

    def _looks_like(self) -> str:
        confusing_sets = [
            "Il1",
            "O0DQ",
            "Z25S",
            "B86G9",
            "nmw",
        ]
        charset = self.rng.choice(confusing_sets)
        length = self.rng.randint(8, 20)
        start_char = self.rng.choice(string.ascii_letters)
        body = "".join(self.rng.choice(charset) for _ in range(length))
        return start_char + body
