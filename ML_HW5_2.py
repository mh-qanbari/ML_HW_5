import numpy as np

g_FILE_DATA = "HW5-2_data_heart.csv"

true = True
false = False

g_BIG_FLOAT = float(1000000.0)


class PatientInfo:
    def __init__(self):
        # Public
        self.features = list()
        self.label = None

    def __getitem__(self, key):
        return self.features[key]

    def __setitem__(self, key, value):
        self.features[key] = value


class PatientsInfo:
    sex = {'male', 'female'}
    cp = set()
    def __init__(self, instance_count, feature_count):
        # age,sex,cp,trestbps,col,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,num
        self.info_list = np.zeros((instance_count, feature_count))
        self.__feature_count = feature_count
        self.__size = 0

    def get_size(self):
        return self.__size, self.__feature_count

    def add_instance(self, feature_list, label=None):
        # pi = PatientInfo()
        # pi.features.append(feature_list)
        # pi.label = label
        # self.info_list.
        # del pi
        self.info_list[self.__size] = feature_list[:]
        self.__size += 1

    def normalize(self):
        # age, *sex, *cp, trestbps, col, chol, *fbs, *restecg, thalach, *exang, oldpeak, *slope, ca, *thal, *num
        age_index = 0
        age_min = g_BIG_FLOAT
        age_max = -1
        trestbps_index = 3
        trestbps_min = g_BIG_FLOAT
        trestbps_max = -1
        col_index = 4
        col_min = g_BIG_FLOAT
        col_max = -1
        chol_index = 5
        chol_min = g_BIG_FLOAT
        chol_max = -1
        thalach_index = 8
        thalach_min = g_BIG_FLOAT
        thalach_max = -1
        oldpeak_index = 10
        oldpeak_min = g_BIG_FLOAT
        oldpeak_max = -1
        ca_index = 12
        ca_min = g_BIG_FLOAT
        ca_max = -1

        for i in range(self.__size):
            age = self.info_list[i, age_index]
            if age is not None:
                if age_min > age: age_min = age
                if age_max < age: age_max = age
            trestbps = self.info_list[i, trestbps_index]
            if trestbps is not None:
                if trestbps_min > trestbps: trestbps_min = trestbps
                if trestbps_max < trestbps: trestbps_max = trestbps
            col = self.info_list[i, col_index]
            if col is not None:
                if col_min > col: col_min = col
                if col_max < col: col_max = col
            chol = self.info_list[i, chol_index]
            if chol is not None:
                if chol_min > chol: chol_min = chol
                if chol_max < chol: chol_max = chol
            thalach = self.info_list[i, thalach_index]
            if thalach is not None:
                if thalach_min > thalach: thalach_min = thalach
                if thalach_max < thalach: thalach_max = thalach
            oldpeak = self.info_list[i, oldpeak_index]
            if oldpeak is not None:
                if oldpeak_min > oldpeak: oldpeak_min = oldpeak
                if oldpeak_max < oldpeak: oldpeak_max = oldpeak
            ca = self.info_list[i, ca_index]
            if ca is not None:
                if ca_min > ca: ca_min = ca
                if ca_max < ca: ca_max = ca
        for i in range(self.__size):
            age = self.info_list[i, age_index]
            if age is not None:
                self.info_list[i, age_index] = (age - age_min) / (age_max - age_min)
            else:
                self.info_list[i, age_index] = 0.5
            trestbps = self.info_list[i, trestbps_index]
            if trestbps is not None:
                self.info_list[i, trestbps_index] = (trestbps - trestbps_min) / (trestbps_max - trestbps_min)
            else:
                self.info_list[i, trestbps_index] = 0.5
            col = self.info_list[i, col_index]
            if col is not None:
                self.info_list[i, col_index] = (col - col_min) / (col_max - col_min)
            else:
                self.info_list[i, col_index] = 0.5
            chol = self.info_list[i, chol_index]
            if chol is not None:
                self.info_list[i, chol_index] = (chol - chol_min) / (chol_max - chol_min)
            else:
                self.info_list[i, chol_index] = 0.5
            thalach = self.info_list[i, thalach_index]
            if thalach is not None:
                self.info_list[i, thalach_index] = (thalach - thalach_min) / (thalach_max - thalach_min)
            else:
                self.info_list[i, thalach_index] = 0.5
            oldpeak = self.info_list[i, oldpeak_index]
            if oldpeak is not None:
                self.info_list[i, oldpeak_index] = (oldpeak - oldpeak_min) / (oldpeak_max - oldpeak_min)
            else:
                self.info_list[i, oldpeak_index] = 0.5
            ca = self.info_list[i, ca_index]
            if ca is not None:
                self.info_list[i, ca_index] = (ca - ca_min) / (ca_max - ca_min)
            else:
                self.info_list[i, ca_index] = 0.5

        sex_index = 1
        x = self.info_list[:, sex_index]
        n = len(x)
        x = FeaturesManager.normalize(x)
        self.info_list[:, sex_index] = np.array(x).reshape(n)
        cp_index = 2
        x = self.info_list[:, cp_index]
        x = FeaturesManager.normalize(x)
        self.info_list[:, cp_index] = np.array(x).reshape(n)
        fbs_index = 6
        x = self.info_list[:, fbs_index]
        x = FeaturesManager.normalize(x)
        self.info_list[:, fbs_index] = np.array(x).reshape(n)
        restecg_index = 7
        x = self.info_list[:, restecg_index]
        x = FeaturesManager.normalize(x)
        self.info_list[:, restecg_index] = np.array(x).reshape(n)
        exang_index = 9
        x = self.info_list[:, exang_index]
        x = FeaturesManager.normalize(x)
        self.info_list[:, exang_index] = np.array(x).reshape(n)
        slope_index = 11
        x = self.info_list[:, slope_index]
        x = FeaturesManager.normalize(x)
        self.info_list[:, slope_index] = np.array(x).reshape(n)
        thal_index = 13
        x = self.info_list[:, thal_index]
        x = FeaturesManager.normalize(x)
        self.info_list[:, thal_index] = np.array(x).reshape(n)
        num_index = 14
        x = self.info_list[:, num_index]
        x = FeaturesManager.normalize(x)
        self.info_list[:, num_index] = np.array(x).reshape(n)


class FeaturesManager:
    def __init__(self):
        pass

    index_sex = 1
    __sex = list()
    @staticmethod
    def sex_value(sex):
        if sex not in FeaturesManager.__sex:
            FeaturesManager.__sex.append(sex)
        return FeaturesManager.__sex.index(sex)

    index_cp = 2
    __cp = list()
    @staticmethod
    def cp_value(cp):
        if cp not in FeaturesManager.__cp:
            FeaturesManager.__cp.append(cp)
        return FeaturesManager.__cp.index(cp)

    index_fbs = 6
    __fbs = list()
    @staticmethod
    def fbs_value(fbs):
        if fbs not in FeaturesManager.__fbs:
            FeaturesManager.__fbs.append(fbs)
        return FeaturesManager.__fbs.index(fbs)

    index_restecg = 7
    __restecg = list()
    @staticmethod
    def restecg_value(restecg):
        if restecg not in FeaturesManager.__restecg:
            FeaturesManager.__restecg.append(restecg)
        return FeaturesManager.__restecg.index(restecg)

    index_exang = 9
    __exang = list()
    @staticmethod
    def exang_value(exang):
        if exang not in FeaturesManager.__exang:
            FeaturesManager.__exang.append(exang)
        return FeaturesManager.__exang.index(exang)

    index_slope = 11
    __slope = list()
    @staticmethod
    def slope_value(slope):
        if slope not in FeaturesManager.__slope:
            FeaturesManager.__slope.append(slope)
        return FeaturesManager.__slope.index(slope)

    index_thal = 13
    __thal = list()
    @staticmethod
    def thal_value(thal):
        if thal not in FeaturesManager.__thal:
            FeaturesManager.__thal.append(thal)
        return FeaturesManager.__thal.index(thal)

    index_num = 14
    __num = list()
    @staticmethod
    def num_value(num):
        if num not in FeaturesManager.__num:
            FeaturesManager.__num.append(num)
        return FeaturesManager.__num.index(num)

    @staticmethod
    def normalize(feature_list, is_nominal=true):
        if is_nominal:
            feature_set = list()
            for feature in feature_list:
                if feature not in feature_set:
                    feature_set.append(feature)
            min_ = 0
            max_ = len(feature_set) - 1
            f_index = -1
            for feature in feature_list:
                f_index += 1
                value = feature_set.index(feature)
                norm_value = (value - min_) / (max_ - min_)
                feature_list[f_index] = norm_value
        else:
            n = len(feature_list)
            min_ = feature_list[0]
            max_ = min_
            for f_index in range(1, n):
                value = feature_list[f_index]
                if min_ > value:
                    min_ = value
                elif max_ < value:
                    max_ = value
            for f_index in range(n):
                value = feature_list[f_index]
                norm_value = (value - min_) / (max_ - min_)
                feature_list[f_index] = norm_value
        return feature_list

if __name__ == "__main__":
    ds = PatientsInfo(303, 15)

    with open(g_FILE_DATA) as file_data:
        ignore_line = true
        for line in file_data:
            if ignore_line:
                ignore_line = false
                continue

            values = line.split(',')
            for v_index in range(len(values)):
                value = values[v_index]
                if value == '?':
                    values[v_index] = None

            cp = values[FeaturesManager.index_cp]
            values[FeaturesManager.index_cp] = FeaturesManager.cp_value(cp)
            exang = values[FeaturesManager.index_exang]
            values[FeaturesManager.index_exang] = FeaturesManager.exang_value(exang)
            fbs = values[FeaturesManager.index_fbs]
            values[FeaturesManager.index_fbs] = FeaturesManager.fbs_value(fbs)
            num = values[FeaturesManager.index_num]
            values[FeaturesManager.index_num] = FeaturesManager.num_value(num)
            restecg = values[FeaturesManager.index_restecg]
            values[FeaturesManager.index_restecg] = FeaturesManager.restecg_value(restecg)
            sex = values[FeaturesManager.index_sex]
            values[FeaturesManager.index_sex] = FeaturesManager.sex_value(sex)
            slope = values[FeaturesManager.index_slope]
            values[FeaturesManager.index_slope] = FeaturesManager.slope_value(slope)
            thal = values[FeaturesManager.index_thal]
            values[FeaturesManager.index_thal] = FeaturesManager.thal_value(thal)

            ds.add_instance(values[:], values[-1])

        print ds.info_list[:][:]
        ds.normalize()
        print "==================="
        print ds.info_list[:][:]
