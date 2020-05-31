import unittest
import numpy as np
from icbs import cut


class MyTestCase(unittest.TestCase):
    def test_cut_no_rem(self):
        img = [
            [[1, 0], [2, 1], [2, 3], [1, 4]],
            [[12, 1], [9, 1], [3, 1], [0, 0]],
            [[1, 4], [1, 2], [1, 3], [1, 5]]
        ]
        img = np.asarray(img)
        img_ret, n_im_per_col = cut(img, n_s_row_max=3, n_s_col_max=2, overlap_w=1, overlap_h=1)
        print("imgs: ", img_ret)
        sol = [
            [
                [[1, 0], [2, 1]],
                [[12, 1], [9, 1]],
                [[1, 4], [1, 2]]
            ],
            [
                [[2, 1], [2, 3]],
                [[9, 1], [3, 1]],
                [[1, 2], [1, 3]]
            ],
            [
                [[2, 3], [1, 4]],
                [[3, 1], [0, 0]],
                [[1, 3], [1, 5]],
            ]
        ]

        sol = np.asarray(sol)
        self.assertEqual(n_im_per_col, 1)
        sol = np.asarray(sol)
        for i in range(len(sol)):
            for j in range(len(sol[i])):
                for k in range(len(sol[i][j])):
                    for l in range(len(sol[i][j][k])):
                        self.assertEqual(img_ret[i][j][k][l], sol[i][j][k][l])

    def test_cut_exceptions(self):
        img = [
            [[1, 0], [2, 1], [2, 3], [1, 4]],
            [[12, 1], [9, 1], [3, 1], [0, 0]],
            [[1, 4], [1, 2], [1, 3], [1, 5]]
        ]
        img = np.asarray(img)
        with self.assertRaises(ValueError):
            cut(img, n_s_row_max=3, n_s_col_max=2, overlap_w=3, overlap_h=1)
        with self.assertRaises(ValueError):
            cut(img, n_s_row_max=3, n_s_col_max=2, overlap_w=4, overlap_h=1)
        with self.assertRaises(ValueError):
            cut(img, n_s_row_max=3, n_s_col_max=2, overlap_w=2, overlap_h=2)
        with self.assertRaises(ValueError):
            cut(img, n_s_row_max=3, n_s_col_max=2, overlap_w=1, overlap_h=10)
        with self.assertRaises(ValueError):
            cut(img, n_s_row_max=3, n_s_col_max=2, overlap_w=3, overlap_h=10)

    def test_cut_rem(self):
        img = [
            [[1, 0], [2, 1], [2, 3]],
            [[12, 1], [9, 1], [3, 1]],
            [[1, 4], [1, 2], [1, 3]],
            [[3, 10], [67, 8], [3, 3]],
            [[1, 12], [2, 4], [0, 65]],
            [[9, 13], [14, 25], [1, 2]],
            [[1, 0], [2, 3], [4, 65]]
        ]
        img = np.asarray(img)
        img_ret, n_im_per_col = cut(img, n_s_row_max=4, n_s_col_max=2, overlap_w=1, overlap_h=3)
        print("imgs: ", img_ret)
        sol = [
            # first row first
            [
                [[1, 0], [2, 1]],
                [[12, 1], [9, 1]],
                [[1, 4], [1, 2]],
                [[3, 10], [67, 8]]
            ],
            [
                [[2, 1], [2, 3]],
                [[9, 1], [3, 1]],
                [[1, 2], [1, 3]],
                [[67, 8], [3, 3]]
            ],
            # second row first
            [
                [[12, 1], [9, 1]],
                [[1, 4], [1, 2]],
                [[3, 10], [67, 8]],
                [[1, 12], [2, 4]]
            ],
            [
                [[9, 1], [3, 1]],
                [[1, 2], [1, 3]],
                [[67, 8], [3, 3]],
                [[2, 4], [0, 65]]
            ],
            # third row first
            [
                [[1, 4], [1, 2]],
                [[3, 10], [67, 8]],
                [[1, 12], [2, 4]],
                [[9, 13], [14, 25]]
            ],
            [
                [[1, 2], [1, 3]],
                [[67, 8], [3, 3]],
                [[2, 4], [0, 65]],
                [[14, 25], [1, 2]]
            ],
            # fourth row first
            [
                [[3, 10], [67, 8]],
                [[1, 12], [2, 4]],
                [[9, 13], [14, 25]],
                [[1, 0], [2, 3]]
            ],
            [
                [[67, 8], [3, 3]],
                [[2, 4], [0, 65]],
                [[14, 25], [1, 2]],
                [[2, 3], [4, 65]]
            ]
        ]

        sol = np.asarray(sol)
        self.assertEqual(n_im_per_col, 4)
        sol = np.asarray(sol)
        for i in range(len(sol)):
            for j in range(len(sol[i])):
                for k in range(len(sol[i][j])):
                    for l in range(len(sol[i][j][k])):
                        self.assertEqual(img_ret[i][j][k][l], sol[i][j][k][l])

    def test_cut_rem_2(self):
        img = [
            [[1, 0], [2, 1], [2, 3]],
            [[12, 1], [9, 1], [3, 1]],
            [[1, 4], [1, 2], [1, 3]],
            [[3, 10], [67, 8], [3, 3]],
            [[1, 12], [2, 4], [0, 65]],
            [[9, 13], [14, 25], [1, 2]],
            [[1, 0], [2, 3], [4, 65]],
            [[9, 1], [4, 2], [7, 2]]
        ]
        img = np.asarray(img)
        img_ret, n_im_per_col = cut(img, n_s_row_max=4, n_s_col_max=2, overlap_w=1, overlap_h=3)
        print("imgs: ", img_ret)
        sol = [
            # first row first
            [
                [[1, 0], [2, 1]],
                [[12, 1], [9, 1]],
                [[1, 4], [1, 2]],
                [[3, 10], [67, 8]]
            ],
            [
                [[2, 1], [2, 3]],
                [[9, 1], [3, 1]],
                [[1, 2], [1, 3]],
                [[67, 8], [3, 3]]
            ],
            # second row first
            [
                [[12, 1], [9, 1]],
                [[1, 4], [1, 2]],
                [[3, 10], [67, 8]],
                [[1, 12], [2, 4]]
            ],
            [
                [[9, 1], [3, 1]],
                [[1, 2], [1, 3]],
                [[67, 8], [3, 3]],
                [[2, 4], [0, 65]]
            ],
            # third row first
            [
                [[1, 4], [1, 2]],
                [[3, 10], [67, 8]],
                [[1, 12], [2, 4]],
                [[9, 13], [14, 25]]
            ],
            [
                [[1, 2], [1, 3]],
                [[67, 8], [3, 3]],
                [[2, 4], [0, 65]],
                [[14, 25], [1, 2]]
            ],
            # fourth row first
            [
                [[3, 10], [67, 8]],
                [[1, 12], [2, 4]],
                [[9, 13], [14, 25]],
                [[1, 0], [2, 3]]
            ],
            [
                [[67, 8], [3, 3]],
                [[2, 4], [0, 65]],
                [[14, 25], [1, 2]],
                [[2, 3], [4, 65]]
            ],
            # fifth row first
            [
                [[1, 12], [2, 4]],
                [[9, 13], [14, 25]],
                [[1, 0], [2, 3]],
                [[9, 1], [4, 2]]
            ],
            [
                [[2, 4], [0, 65]],
                [[14, 25], [1, 2]],
                [[2, 3], [4, 65]],
                [[4, 2], [7, 2]]
            ]
        ]

        sol = np.asarray(sol)
        sol = np.asarray(sol)
        for i in range(len(sol)):
            for j in range(len(sol[i])):
                for k in range(len(sol[i][j])):
                    for l in range(len(sol[i][j][k])):
                        self.assertEqual(img_ret[i][j][k][l], sol[i][j][k][l])
        self.assertEqual(5, n_im_per_col)

    def test_cut_rem_3(self):
        img = [
            [[1, 0], [2, 1], [2, 3]],
            [[12, 1], [9, 1], [3, 1]],
            [[1, 4], [1, 2], [1, 3]],
            [[3, 10], [67, 8], [3, 3]],
            [[1, 12], [2, 4], [0, 65]],
            [[9, 13], [14, 25], [1, 2]]
        ]
        img = np.asarray(img)
        img_ret, dims = cut(img, n_s_row_max=4, n_s_col_max=2, overlap_w=1, overlap_h=3)
        print("dims: ", dims)
        print("imgs: ", img_ret)
        sol = [
            # first row first
            [
                [[1, 0], [2, 1]],
                [[12, 1], [9, 1]],
                [[1, 4], [1, 2]],
                [[3, 10], [67, 8]]
            ],
            [
                [[2, 1], [2, 3]],
                [[9, 1], [3, 1]],
                [[1, 2], [1, 3]],
                [[67, 8], [3, 3]]
            ],
            # second row first
            [
                [[12, 1], [9, 1]],
                [[1, 4], [1, 2]],
                [[3, 10], [67, 8]],
                [[1, 12], [2, 4]]
            ],
            [
                [[9, 1], [3, 1]],
                [[1, 2], [1, 3]],
                [[67, 8], [3, 3]],
                [[2, 4], [0, 65]]
            ],
            # third row first
            [
                [[1, 4], [1, 2]],
                [[3, 10], [67, 8]],
                [[1, 12], [2, 4]],
                [[9, 13], [14, 25]]
            ],
            [
                [[1, 2], [1, 3]],
                [[67, 8], [3, 3]],
                [[2, 4], [0, 65]],
                [[14, 25], [1, 2]]
            ]
        ]

        sol = np.asarray(sol)
        sol = np.asarray(sol)
        for i in range(len(sol)):
            for j in range(len(sol[i])):
                for k in range(len(sol[i][j])):
                    for l in range(len(sol[i][j][k])):
                        self.assertEqual(img_ret[i][j][k][l], sol[i][j][k][l])

    def test_cut_rem_4(self):
        img = [
            [[1, 0], [2, 1], [2, 3]],
            [[12, 1], [9, 1], [3, 1]],
            [[1, 4], [1, 2], [1, 3]],
            [[3, 10], [67, 8], [3, 3]],
            [[1, 12], [2, 4], [0, 65]],
            [[9, 13], [14, 25], [1, 2]],
            [[1, 0], [2, 3], [4, 65]],
            [[9, 1], [4, 2], [7, 2]]
        ]
        img = np.asarray(img)
        img_ret, n_im_per_col = cut(img, n_s_row_max=4, n_s_col_max=2, overlap_w=1, overlap_h=1)
        print("imgs: ", img_ret)
        sol = [
            # first row first
            [
                [[1, 0], [2, 1]],
                [[12, 1], [9, 1]],
                [[1, 4], [1, 2]],
                [[3, 10], [67, 8]]
            ],
            [
                [[2, 1], [2, 3]],
                [[9, 1], [3, 1]],
                [[1, 2], [1, 3]],
                [[67, 8], [3, 3]]
            ],
            # fourth row first
            [
                [[3, 10], [67, 8]],
                [[1, 12], [2, 4]],
                [[9, 13], [14, 25]],
                [[1, 0], [2, 3]]
            ],
            [
                [[67, 8], [3, 3]],
                [[2, 4], [0, 65]],
                [[14, 25], [1, 2]],
                [[2, 3], [4, 65]]
            ],
            # fifth row first
            [
                [[1, 0], [2, 3]],
                [[9, 1], [4, 2]]
            ],
            [
                [[2, 3], [4, 65]],
                [[4, 2], [7, 2]]
            ]
        ]
        sol = np.asarray(sol)
        for i in range(len(sol)):
            for j in range(len(sol[i])):
                for k in range(len(sol[i][j])):
                    for l in range(len(sol[i][j][k])):
                        self.assertEqual(img_ret[i][j][k][l], sol[i][j][k][l])
        self.assertEqual(3, n_im_per_col)

    def test_cut_rem_5(self):
        img = [
            [[1, 0], [2, 1], [2, 3], [4, 5]],
            [[12, 1], [9, 1], [3, 1], [6, 7]],
            [[1, 4], [1, 2], [1, 3], [7, 9]],
            [[3, 10], [67, 8], [3, 3], [13, 4]],
            [[1, 12], [2, 4], [0, 65], [11, 2]],
            [[9, 13], [14, 25], [1, 2], [3, 2]],
            [[1, 0], [2, 3], [4, 65], [6, 7]],
            [[9, 1], [4, 2], [7, 2], [8, 9]]
        ]
        img = np.asarray(img)
        img_ret, n_im_per_col = cut(img, n_s_row_max=4, n_s_col_max=4, overlap_w=3, overlap_h=1)
        print("imgs: ", img_ret)
        sol = [
            # first row first
            [
                [[1, 0], [2, 1], [2, 3], [4, 5]],
                [[12, 1], [9, 1], [3, 1], [6, 7]],
                [[1, 4], [1, 2], [1, 3], [7, 9]],
                [[3, 10], [67, 8], [3, 3], [13, 4]]
            ],
            [
                [[3, 10], [67, 8], [3, 3], [13, 4]],
                [[1, 12], [2, 4], [0, 65], [11, 2]],
                [[9, 13], [14, 25], [1, 2], [3, 2]],
                [[1, 0], [2, 3], [4, 65], [6, 7]]
            ],
            [
                [[1, 0], [2, 3], [4, 65], [6, 7]],
                [[9, 1], [4, 2], [7, 2], [8, 9]]
            ]
        ]
        sol = np.asarray(sol)
        for i in range(len(sol)):
            for j in range(len(sol[i])):
                for k in range(len(sol[i][j])):
                    for l in range(len(sol[i][j][k])):
                        self.assertEqual(img_ret[i][j][k][l], sol[i][j][k][l])
        self.assertEqual(3, n_im_per_col)

    def test_cut_rem_6(self):
        img = [
            [[1, 0], [2, 1], [2, 3], [4, 5], [3, 5]],
            [[12, 1], [9, 1], [3, 1], [6, 7], [1, 2]],
            [[1, 4], [1, 2], [1, 3], [7, 9], [4, 8]],
            [[3, 10], [67, 8], [3, 3], [13, 4], [3, 6]],
            [[1, 12], [2, 4], [0, 65], [11, 2], [7, 3]],
            [[9, 13], [14, 25], [1, 2], [3, 2], [0, 1]],
            [[1, 0], [2, 3], [4, 65], [6, 7], [17, 81]],
            [[9, 1], [4, 2], [7, 2], [8, 9], [44, 10]]
        ]
        img = np.asarray(img)
        img_ret, n_im_per_col = cut(img, n_s_row_max=4, n_s_col_max=4, overlap_w=1, overlap_h=1)
        print("imgs: ", img_ret)
        sol = [
            # first row first
            [
                [[1, 0], [2, 1], [2, 3], [4, 5]],
                [[12, 1], [9, 1], [3, 1], [6, 7]],
                [[1, 4], [1, 2], [1, 3], [7, 9]],
                [[3, 10], [67, 8], [3, 3], [13, 4]]
            ],
            [
                [[4, 5], [3, 5]],
                [[6, 7], [1, 2]],
                [[7, 9], [4, 8]],
                [[13, 4], [3, 6]]
            ],
            # fourth row first
            [
                [[3, 10], [67, 8], [3, 3], [13, 4]],
                [[1, 12], [2, 4], [0, 65], [11, 2]],
                [[9, 13], [14, 25], [1, 2], [3, 2]],
                [[1, 0], [2, 3], [4, 65], [6, 7]]
            ],
            [
                [[13, 4], [3, 6]],
                [[11, 2], [7, 3]],
                [[3, 2], [0, 1]],
                [[6, 7], [17, 81]]
            ],
            # seventh row first
            [
                [[1, 0], [2, 3], [4, 65], [6, 7]],
                [[9, 1], [4, 2], [7, 2], [8, 9]]
            ],
            [
                [[6, 7], [17, 81]],
                [[8, 9], [44, 10]]
            ]
        ]

        sol = np.asarray(sol)
        for i in range(len(sol)):
            for j in range(len(sol[i])):
                for k in range(len(sol[i][j])):
                    for l in range(len(sol[i][j][k])):
                        self.assertEqual(img_ret[i][j][k][l], sol[i][j][k][l])
        self.assertEqual(3, n_im_per_col)

    def test_cut_rem_7(self):
        img = [
            [[1, 0], [2, 1], [2, 3], [4, 5], [3, 5]],
            [[12, 1], [9, 1], [3, 1], [6, 7], [1, 2]],
            [[1, 4], [1, 2], [1, 3], [7, 9], [4, 8]],
            [[3, 10], [67, 8], [3, 3], [13, 4], [3, 6]],
            [[1, 12], [2, 4], [0, 65], [11, 2], [7, 3]],
            [[9, 13], [14, 25], [1, 2], [3, 2], [0, 1]],
            [[1, 0], [2, 3], [4, 65], [6, 7], [17, 81]],
            [[9, 1], [4, 2], [7, 2], [8, 9], [44, 10]]
        ]
        img = np.asarray(img)
        # overlap_h = 2 overlap_w = 2
        img_ret, n_im_per_col = cut(img, n_s_row_max=4, n_s_col_max=4)
        print("imgs: ", img_ret)
        sol = [
            # first row first
            [
                [[1, 0], [2, 1], [2, 3], [4, 5]],
                [[12, 1], [9, 1], [3, 1], [6, 7]],
                [[1, 4], [1, 2], [1, 3], [7, 9]],
                [[3, 10], [67, 8], [3, 3], [13, 4]]
            ],
            [
                [[2, 3], [4, 5], [3, 5]],
                [[3, 1], [6, 7], [1, 2]],
                [[1, 3], [7, 9], [4, 8]],
                [[3, 3], [13, 4], [3, 6]]
            ],
            # third row first
            [
                [[1, 4], [1, 2], [1, 3], [7, 9]],
                [[3, 10], [67, 8], [3, 3], [13, 4]],
                [[1, 12], [2, 4], [0, 65], [11, 2]],
                [[9, 13], [14, 25], [1, 2], [3, 2]]
            ],
            [
                [[1, 3], [7, 9], [4, 8]],
                [[3, 3], [13, 4], [3, 6]],
                [[0, 65], [11, 2], [7, 3]],
                [[1, 2], [3, 2], [0, 1]],
            ],
            # fifth row first
            [
                [[1, 12], [2, 4], [0, 65], [11, 2]],
                [[9, 13], [14, 25], [1, 2], [3, 2]],
                [[1, 0], [2, 3], [4, 65], [6, 7]],
                [[9, 1], [4, 2], [7, 2], [8, 9]]
            ],
            [
                [[0, 65], [11, 2], [7, 3]],
                [[1, 2], [3, 2], [0, 1]],
                [[4, 65], [6, 7], [17, 81]],
                [[7, 2], [8, 9], [44, 10]]
            ]
        ]

        sol = np.asarray(sol)
        for i in range(len(sol)):
            for j in range(len(sol[i])):
                for k in range(len(sol[i][j])):
                    for l in range(len(sol[i][j][k])):
                        self.assertEqual(img_ret[i][j][k][l], sol[i][j][k][l])
        self.assertEqual(3, n_im_per_col)

    if __name__ == '__main__':
        unittest.main()
