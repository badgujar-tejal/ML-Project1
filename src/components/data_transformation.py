import sys
import os
import numpy as np
import pandas as pd

from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomeException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessing_obj_file_path = os.path.join('artifacts','preprocessor.pkl')
    