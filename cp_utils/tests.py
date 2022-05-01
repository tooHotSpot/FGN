import unittest
from time import sleep

from cp_dir_file_ops import *
from cp_time import *


class MyTestCase(unittest.TestCase):
    def test_test_dir_created(self):
        self.assertTrue(os.path.exists('tests'),
                        msg='Dir for tests file does not exist')

    def test_dir_exists(self):
        dir_to_create = 'tests/MyWay'
        os.mkdir(dir_to_create)
        self.assertTrue(check_dir_if_exists(dir_to_create))
        shutil.rmtree(dir_to_create)

    def test_illegal_sings_unsafe_method(self):
        t = datetime_log()
        self.assertTrue(':' in t)

    def test_illegal_sings_in_safe_method(self):
        t = datetime_log_fancy()
        self.assertTrue(':' not in t)

    def test_dir_not_exists(self):
        dir_to_check = datetime_log_fancy()
        self.assertFalse(check_dir_if_exists(dir_to_check))

    def test_create_dir_with_remove(self):
        # Test preparation
        dir_to_create = 'upper_with_remove'
        dir_to_create_fp = os.path.join('tests', dir_to_create)
        os.mkdir(dir_to_create_fp)
        subdir_to_create = 'lower_with_remove'
        subdir_to_create_fp = os.path.join('tests', dir_to_create, subdir_to_create)
        os.mkdir(subdir_to_create_fp)
        # Test body
        create_empty_dir_unsafe(dir_to_create_fp)
        self.assertTrue(os.path.exists(dir_to_create_fp))
        self.assertFalse(os.path.exists(subdir_to_create_fp))
        shutil.rmtree(dir_to_create_fp)

    def test_create_dir_no_remove_allowed_absent(self):
        # Test preparation
        dir_to_create = 'upper_allowed_empty_absent'
        dir_to_create_fp = os.path.join('tests', dir_to_create)
        # Test body
        create_empty_dir_safe(dir_to_create_fp)
        self.assertTrue(os.path.exists(dir_to_create_fp))
        shutil.rmtree(dir_to_create_fp)

    def test_create_dir_no_remove_allowed_empty(self):
        # Test preparation
        dir_to_create = 'upper_allowed_empty'
        dir_to_create_fp = os.path.join('tests', dir_to_create)
        os.mkdir(dir_to_create_fp)
        # Test body
        create_empty_dir_safe(dir_to_create_fp)
        self.assertTrue(os.path.exists(dir_to_create_fp))
        shutil.rmtree(dir_to_create_fp)

    def test_create_dir_no_remove_allowed_not_empty(self):
        # Test preparation
        dir_to_create = 'upper_allowed_not_empty'
        dir_to_create_fp = os.path.join('tests', dir_to_create)
        os.mkdir(dir_to_create_fp)
        subdir_to_create = 'lower_allowed_not_empty'
        subdir_to_create_fp = os.path.join('tests', dir_to_create, subdir_to_create)
        os.mkdir(subdir_to_create_fp)
        # Test body
        create_empty_dir_safe(dir_to_create_fp)
        self.assertTrue(os.path.exists(subdir_to_create_fp))
        shutil.rmtree(dir_to_create_fp)

    def test_no_msces_datetime(self):
        a = datetime_now()
        b = datetime_now()
        result = datetime_diff(a, b)
        self.assertFalse('mcsec' in result)

    def test_no_msces_time(self):
        a = datetime_now()
        b = datetime_now()
        result = datetime_diff(a, b)
        self.assertFalse('mcsec' in result)

    def test_msces_in_datetime(self):
        a = datetime_now()
        sleep(0.8)
        b = datetime_now()
        result = datetime_diff(a, b)
        self.assertTrue('mcsec' in result)

    def test_msces_in_time(self):
        a = datetime_now()
        sleep(0.8)
        b = datetime_now()
        result = datetime_diff(a, b)
        self.assertTrue('mcsec' in result)

    def test_check_notebook_pc(self):
        if os.name == 'nt' and os.path.exists('C:\\Users\\Art'):
            self.assertEqual(define_env(), 'PC')
        else:
            self.assertEqual(define_env(), 'OTHER')

    def test_write_json_unsafe_file_absent(self):
        data_write = {'Some': 'Text'}
        file_sp = time_log_fancy() + \
                  '_write_json_unsafe.json'
        file_fp = os.path.join('tests', file_sp)
        write_json_safe(file_fp, data_write)
        self.assertTrue(check_file_if_exists(file_fp))
        os.remove(file_fp)

    def test_write_pkl_unsafe_file_absent(self):
        data_write = {'Some': 'Text'}
        file_sp = time_log_fancy() + \
                  '_write_pkl_unsafe.pkl'
        file_fp = os.path.join('tests', file_sp)
        write_pkl_safe(file_fp, data_write)
        self.assertTrue(check_file_if_exists(file_fp))
        os.remove(file_fp)

    def test_write_json_unsafe_file_present(self):
        data_write = {'Some': 'Text'}
        file_sp = time_log_fancy() + \
                  '_write_json_unsafe_file_absent.json'
        file_fp = os.path.join('tests', file_sp)
        write_json_safe(file_fp, data_write)
        data_write_new = {'Some': 'Song'}
        write_json_unsafe(file_fp, data_write_new)
        data_read = read_json(file_fp)
        self.assertTrue(data_read == data_write_new)
        os.remove(file_fp)

    def test_write_pkl_unsafe_file_present(self):
        data_write = {'Some': 'Text'}
        file_sp = time_log_fancy() + \
                  '_write_pkl_unsafe_file_absent.pkl'
        file_fp = os.path.join('tests', file_sp)
        write_pkl_safe(file_fp, data_write)
        data_write_new = {'Some': 'Song'}
        write_pkl_unsafe(file_fp, data_write_new)
        data_read = read_pkl(file_fp)
        self.assertTrue(data_read == data_write_new)
        os.remove(file_fp)

    def test_read_json_file_present(self):
        data_write = {'Some': 'Text'}
        file_sp = time_log_fancy() + \
                  '_reading_json.json'
        file_fp = os.path.join('tests', file_sp)
        write_json_unsafe(file_fp, data_write)
        data_read = read_json(file_fp)
        self.assertTrue(data_read == data_write)
        os.remove(file_fp)

    def test_read_pkl_file_present(self):
        data_write = {'Some': 'Text'}
        file_sp = time_log_fancy() + \
                  '_reading_pkl_present.pkl'
        file_fp = os.path.join('tests', file_sp)
        write_pkl_unsafe(file_fp, data_write)
        data_read = read_pkl(file_fp)
        self.assertTrue(data_read == data_write)
        os.remove(file_fp)

    def test_read_json_file_absent(self):
        file_sp = time_log_fancy() + \
                  '_reading_json.json'
        file_fp = os.path.join('tests', file_sp)
        data_read = read_json(file_fp)
        self.assertTrue(len(data_read) == 0)

    def test_read_pkl_file_absent(self):
        file_sp = time_log_fancy() + \
                  '_reading_pkl_absent.pkl'
        file_fp = os.path.join('tests', file_sp)
        data_read = read_pkl(file_fp)
        self.assertTrue(len(data_read) == 0)

    def test_write_json_safe_file_absent(self):
        data_write = {'Some': 'Text'}
        file_sp = time_log_fancy() + \
                  '_write_json_safe_file_absent.json'
        file_fp = os.path.join('tests', file_sp)
        write_json_safe(file_fp, data_write)
        data_write_new = {'Some': 'Song'}
        write_json_safe(file_fp, data_write_new)
        data_read = read_json(file_fp)
        self.assertTrue(data_read == data_write)
        os.remove(file_fp)

    def test_write_pkl_safe_file_absent(self):
        data_write = {'Some': 'Text'}
        file_sp = time_log_fancy() + \
                  '_write_pkl_safe_file_absent.pkl'
        file_fp = os.path.join('tests', file_sp)
        write_pkl_safe(file_fp, data_write)
        data_write_new = {'Some': 'Song'}
        write_pkl_safe(file_fp, data_write_new)
        data_read = read_pkl(file_fp)
        self.assertTrue(data_read == data_write)
        os.remove(file_fp)


if __name__ == '__main__':
    unittest.main()
