# !/usr/bin/env python
# -*- coding: utf-8 -*-

import jieba.analyse
from math import sqrt
from functools import reduce

class Similarity(object):
	"""docstring for Similarity"""
	def __init__(self, src, dis, topK = 20):
		super(Similarity, self).__init__()
		self._src = src
		self._dis = dis
		self._topK = topK

	# 余弦定理实现
	def cosSim(self):
		self._src_dict = {}
		self._dis_dict = {}
		src_topK = jieba.analyse.extract_tags(self._src, topK = self._topK, withWeight=True)
		dis_topK = jieba.analyse.extract_tags(self._dis, topK = self._topK, withWeight=True)
		for k, v in src_topK:
			self._src_dict[k] = v
		for k, v in dis_topK:
			self._dis_dict[k] = v

		for k in self._src_dict:
			self._dis_dict[k] = self._dis_dict.get(k, 0)
		for k in self._dis_dict:
			self._src_dict[k] = self._src_dict.get(k, 0)
		def relative(dicts):
			# 计算相对词频率
			_max = max(dicts.values())
			_min = min(dicts.values())
			_mid = _max - _min
			for k in dicts:
				dicts[k] = (dicts[k] - _min)/_mid
			return dicts
		self._src_dict = relative(self._src_dict)
		self._dis_dict = relative(self._dis_dict)

		total = 0
		for k in self._src_dict:
			total += self._src_dict[k] * self._dis_dict[k]
		A = sqrt(reduce(lambda x, y: x+y, map(lambda x: x*x, self._src_dict.values())))
		B = sqrt(reduce(lambda x, y: x+y, map(lambda x: x*x, self._dis_dict.values())))
		return total/(A*B)

def main():
	t1 = 'I love China'
	t2 = 'I love Chinese'
	sim = Similarity(t1, t2)
	print(sim.cosSim())
if __name__ == '__main__':
	main()