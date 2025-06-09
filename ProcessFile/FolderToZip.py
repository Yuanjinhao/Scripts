# -*- coding:utf-8 -*-
"""
@Time:2024/3/4  10:07
@Auth:yuanjinhao
@File:FolderToZip.py
"""
import zipfile
import os
from tqdm import tqdm


def folder_to_zip(folderpath, zipath):
	zip = zipfile.ZipFile(zipath, 'w', zipfile.ZIP_DEFLATED)
	for path, dirnames, filenames in os.walk(folderpath):
		fpath = path.replace(os.path.dirname(folderpath), '')
		for filename in filenames:
			zip.write(os.path.join(path, filename), os.path.join(fpath, filename))
	zip.close()


if __name__ == "__main__":
	root = os.getcwd()
	root_name = 'LightAttributeDatasets'
	dirs_name = os.listdir(root_name)
	for dir in tqdm(dirs_name):
		folderpath = root + '\\' + root_name + '\\' + dir + '\\regionOverlapSize' + '\\Data'
		if dir[:11] == 'Cleaning_20':
			zipath = root + '\\' + root_name + '\\' + dir[:8] + 'Data' + dir[8:] + '.zip'
		else:
			zipath = root + '\\' + root_name + '\\' + dir + '.zip'
		folder_to_zip(folderpath, zipath)
