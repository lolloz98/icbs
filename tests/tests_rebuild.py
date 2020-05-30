import unittest
from rebuild import *
from cut import cut_img_pro, get_default_overlaps
import spectral.io.envi as envi


class MyTestCase(unittest.TestCase):

    def test_mean(self):
        self.assertAlmostEqual(mean([1, 2, 3]), 2, delta=0.00002)
        self.assertAlmostEqual(mean([1, 2.5, 3]), 2.166666666, delta=0.00002)
        self.assertAlmostEqual(mean([1, 2, 3, 10, 0]), 3.2, delta=0.00002)

    def test_reorder_cut_img(self):
        # in test_cut_img_pro_no_rem of tests_cut_img_pro.py
        img = [
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

        img_ret = reorder_cut_img(img, 1)

        sol = [
            [
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
        ]

        for i in range(len(sol)):
            for j in range(len(sol[i])):
                for k in range(len(sol[i][j])):
                    for l in range(len(sol[i][j][k])):
                        for d in range(len(sol[i][j][k][l])):
                            self.assertEqual(img_ret[i][j][k][l][d], sol[i][j][k][l][d])

    def test_reorder_cut_img_1(self):
        # in test_cut_img_pro_rem of tests_cut_img_pro.py
        img = [
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

        n_im_per_col = 4
        img_ret = reorder_cut_img(img, n_im_per_col)

        sol = [
            # first row first
            [
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
                ]
            ],
            # second row first
            [
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
                ]
            ],
            # third row first
            [
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
            ,
            # fourth row first
            [
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
        ]

        for i in range(len(sol)):
            for j in range(len(sol[i])):
                for k in range(len(sol[i][j])):
                    for l in range(len(sol[i][j][k])):
                        for d in range(len(sol[i][j][k][l])):
                            self.assertEqual(img_ret[i][j][k][l][d], sol[i][j][k][l][d])

    def test_reorder_cut_img_2(self):
        # in test_cut_img_pro_rem_6 of tests_cut_img_pro.py
        img = [
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

        n_im_per_col = 3
        img_ret = reorder_cut_img(img, n_im_per_col)

        sol = [
            # first row first
            [
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
                ]
            ],
            # fourth row first
            [
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
                ]
            ],
            # seventh row first
            [
                [
                    [[1, 0], [2, 3], [4, 65], [6, 7]],
                    [[9, 1], [4, 2], [7, 2], [8, 9]]
                ],
                [
                    [[6, 7], [17, 81]],
                    [[8, 9], [44, 10]]
                ]
            ]
        ]

        for i in range(len(sol)):
            for j in range(len(sol[i])):
                for k in range(len(sol[i][j])):
                    for l in range(len(sol[i][j][k])):
                        for d in range(len(sol[i][j][k][l])):
                            self.assertEqual(img_ret[i][j][k][l][d], sol[i][j][k][l][d])

    def test_step1_to_rebuild(self):
        # from test_reorder_cut_img
        img = [
            [
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
        ]

        step1 = step1_to_rebuild(img, n_s_row_max=3, n_s_col_max=2, overlap_w=1, overlap_h=1, W=4, H=3)

        sol = [
                [
                    [[1, 0]], [[2, 1], [2, 1]], [[2, 3], [2, 3]], [[1, 4]]
                ],
                [
                    [[12, 1]], [[9, 1], [9, 1]], [[3, 1], [3, 1]], [[0, 0]]
                ],
                [
                    [[1, 4]], [[1, 2], [1, 2]], [[1, 3], [1, 3]], [[1, 5]]
                ]
        ]
        for i in range(len(sol)):
            for j in range(len(sol[i])):
                for k in range(len(sol[i][j])):
                    for l in range(len(sol[i][j][k])):
                        self.assertEqual(step1[i][j][k][l], sol[i][j][k][l])

    def test_step1_to_rebuild_1(self):
        # from test_reorder_cut_img_1
        img = [
            # first row first
            [
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
                ]
            ],
            # second row first
            [
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
                ]
            ],
            # third row first
            [
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
            ,
            # fourth row first
            [
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
        ]

        step1 = step1_to_rebuild(img, n_s_row_max=4, n_s_col_max=2, overlap_w=1, overlap_h=3, W=3, H=7)

        sol = [
            [
                [[1, 0]], [[2, 1], [2, 1]], [[2, 3]]
            ],
            [
                [[12, 1], [12, 1]], [[9, 1], [9, 1], [9, 1], [9, 1]], [[3, 1], [3, 1]]
            ],
            [
                [[1, 4], [1, 4], [1, 4]], [[1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]], [[1, 3], [1, 3], [1, 3]]
            ],
            [
                [[3, 10], [3, 10], [3, 10], [3, 10]], [[67, 8], [67, 8], [67, 8], [67, 8], [67, 8], [67, 8], [67, 8], [67, 8]], [[3, 3], [3, 3], [3, 3], [3, 3]]
            ],
            [
                [[1, 12], [1, 12], [1, 12]], [[2, 4], [2, 4], [2, 4], [2, 4], [2, 4], [2, 4]], [[0, 65], [0, 65], [0, 65]]
            ],
            [
                [[9, 13], [9, 13]], [[14, 25], [14, 25], [14, 25], [14, 25]], [[1, 2], [1, 2]]
            ],
            [
                [[1, 0]], [[2, 3], [2, 3]], [[4, 65]]
            ]
        ]

        print(step1)

        for i in range(len(step1)):
            for j in range(len(step1[i])):
                for k in range(len(step1[i][j])):
                    for l in range(len(step1[i][j][k])):
                        self.assertEqual(step1[i][j][k][l], sol[i][j][0][l])

    def test_final_step_rebuild(self):
        img = [
                [
                    [[1, 0]], [[2, 1], [2, 1]], [[2, 3], [2, 3]], [[1, 4]]
                ],
                [
                    [[12, 1]], [[9, 1], [9, 1]], [[3, 1], [3, 1]], [[0, 0]]
                ],
                [
                    [[1, 4]], [[1, 2], [1, 2]], [[1, 3], [1, 3]], [[1, 5]]
                ]
        ]

        img_ret = final_step_rebuild(img)

        sol = [
            [[1, 0], [2, 1], [2, 3], [1, 4]],
            [[12, 1], [9, 1], [3, 1], [0, 0]],
            [[1, 4], [1, 2], [1, 3], [1, 5]]
        ]

        self.assertEqual(img_ret, sol)
        for i in range(len(sol)):
            for j in range(len(sol[i])):
                for k in range(len(sol[i][j])):
                    test = sol[i][j][k]
                    test2 = img_ret[i][j][k]
                    self.assertEqual(img_ret[i][j][k], sol[i][j][k])

    def test_final_step_rebuild_1(self):
        img = [
            [
                [[1, 0]], [[2, 1], [2, 1]], [[2, 3]]
            ],
            [
                [[12, 1], [12, 1]], [[9, 1], [9, 1], [9, 1], [9, 1]], [[3, 1], [3, 1]]
            ],
            [
                [[1, 4], [1, 4], [1, 4]], [[1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]], [[1, 3], [1, 3], [1, 3]]
            ],
            [
                [[3, 10], [3, 10], [3, 10], [3, 10]], [[67, 8], [67, 8], [67, 8], [67, 8], [67, 8], [67, 8], [67, 8], [67, 8]], [[3, 3], [3, 3], [3, 3], [3, 3]]
            ],
            [
                [[1, 12], [1, 12], [1, 12]], [[2, 4], [2, 4], [2, 4], [2, 4], [2, 4], [2, 4]], [[0, 65], [0, 65], [0, 65]]
            ],
            [
                [[9, 13], [9, 13]], [[14, 25], [14, 25], [14, 25], [14, 25]], [[1, 2], [1, 2]]
            ],
            [
                [[1, 0]], [[2, 3], [2, 3]], [[4, 65]]
            ]
        ]

        img_ret = final_step_rebuild(img)

        sol = [
            [[1, 0], [2, 1], [2, 3]],
            [[12, 1], [9, 1], [3, 1]],
            [[1, 4], [1, 2], [1, 3]],
            [[3, 10], [67, 8], [3, 3]],
            [[1, 12], [2, 4], [0, 65]],
            [[9, 13], [14, 25], [1, 2]],
            [[1, 0], [2, 3], [4, 65]]
        ]

        self.assertEqual(img_ret, sol)
        for i in range(len(sol)):
            for j in range(len(sol[i])):
                for k in range(len(sol[i][j])):
                    test = sol[i][j][k]
                    test2 = img_ret[i][j][k]
                    self.assertEqual(img_ret[i][j][k], sol[i][j][k])

    def test_effective_mean(self):
        img = [
            [
                [[1, 0]], [[4, 2], [2, 1]], [[2, 3], [2, 3]], [[1, 4]]
            ],
            [
                [[12, 1]], [[9, 1], [9, 1]], [[7, 1], [3, 1]], [[0, 0]]
            ],
            [
                [[1, 4]], [[1, 2], [1, 3]], [[1, 3], [1, 3]], [[1, 5]]
            ]
        ]

        img_ret = final_step_rebuild(img)

        sol = [
            [[1, 0], [3, 1.5], [2, 3], [1, 4]],
            [[12, 1], [9, 1], [5, 1], [0, 0]],
            [[1, 4], [1, 2.5], [1, 3], [1, 5]]
        ]

        self.assertEqual(img_ret, sol)
        for i in range(len(sol)):
            for j in range(len(sol[i])):
                for k in range(len(sol[i][j])):
                    test = sol[i][j][k]
                    test2 = img_ret[i][j][k]
                    self.assertEqual(img_ret[i][j][k], sol[i][j][k])

    def test_effective_mean_1(self):
        img = [
            [
                [[1, 0]], [[2, 1], [2, 1]], [[2, 3]]
            ],
            [
                [[12, 1], [12, 1]], [[9, 1], [9, 5], [9, 1], [9, 1]], [[3, 1], [3, 1]]
            ],
            [
                [[1, 4], [1, 4], [1, 4]], [[1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]], [[1, 3], [1, 3], [1, 3]]
            ],
            [
                [[3, 10], [3, 10], [3, 10], [3, 10]], [[67, 8], [67, 8], [67, 8], [67, 8], [67, 8], [67, 8], [67, 8], [67, 8]], [[3, 3], [3, 3], [3, 3], [3, 3]]
            ],
            [
                [[1, 12], [4, 12], [1, 12]], [[2, 4], [2, 4], [2, 4], [2, 4], [2, 4], [2, 4]], [[0, 65], [0, 65], [0, 65]]
            ],
            [
                [[9, 13], [9, 13]], [[18, 25], [14, 25], [14, 25], [14, 25]], [[4, 3], [1, 2]]
            ],
            [
                [[1, 1]], [[2, 3], [2, 3]], [[4, 65]]
            ]
        ]

        img_ret = final_step_rebuild(img)

        sol = [
            [[1, 0], [2, 1], [2, 3]],
            [[12, 1], [9, 2], [3, 1]],
            [[1, 4], [1, 2], [1, 3]],
            [[3, 10], [67, 8], [3, 3]],
            [[2, 12], [2, 4], [0, 65]],
            [[9, 13], [15, 25], [2.5, 2.5]],
            [[1, 1], [2, 3], [4, 65]]
        ]

        self.assertEqual(img_ret, sol)
        for i in range(len(sol)):
            for j in range(len(sol[i])):
                for k in range(len(sol[i][j])):
                    test = sol[i][j][k]
                    test2 = img_ret[i][j][k]
                    self.assertEqual(img_ret[i][j][k], sol[i][j][k])

    def test_rebuild(self):
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
        img_cut, n_im_per_col = cut_img_pro(img, n_s_row_max=4, n_s_col_max=2, overlap_w=1, overlap_h=3)
        img_reb = rebuild(img_cut, n_im_per_col, n_s_row_max=4, n_s_col_max=2, overlap_w=1, overlap_h=3, W=3, H=7)

        for i in range(len(img)):
            for j in range(len(img[i])):
                for k in range(len(img[i][j])):
                    test = img[i][j][k]
                    test2 = img_reb[i][j][k]
                    self.assertAlmostEqual(img_reb[i][j][k], img[i][j][k], delta=0.000001)

    def test_rebuild_1(self):

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
        img_cut, n_im_per_col = cut_img_pro(img, n_s_row_max=4, n_s_col_max=4)

        img_reb = rebuild(img_cut, n_im_per_col, n_s_row_max=4, n_s_col_max=4, overlap_w=2, overlap_h=2, W=5, H=8)

        for i in range(len(img)):
            for j in range(len(img[i])):
                for k in range(len(img[i][j])):
                    test = img[i][j][k]
                    test2 = img_reb[i][j][k]
                    self.assertAlmostEqual(img_reb[i][j][k], img[i][j][k], delta=0.000001)

    if __name__ == '__main__':
        unittest.main()
