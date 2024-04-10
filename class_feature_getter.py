from typing import Any

class FeatureGetter:
	def __init__(self, params: list = []) -> None:
		self.__params = params

	def __call__(self, token, idx) -> Any:
		pass

	def update_params(self, params: list) -> None:
		self.__params = params

	def get_params(self) -> list:
		return self.__params