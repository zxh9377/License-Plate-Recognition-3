# encoding: utf-8
"""
@author: yangguang
@contact: 40334361@qq.com
@file: main.py
@time: 2019/4/24 19:28
"""
import sys
import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split

"""
    使用方法：
    · 训练时，使626行 train_flag = 1
    · 测试时，使626行 train_flag = 0
    然后直接运行
"""

# os.environ['CUDA_VISIBLE_DEVICES'] = '6'

numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

alphbets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z']

chinese = ['川', '鄂', '赣', '甘', '贵', '桂', '黑', '沪', '冀', '津',
           '京', '吉', '辽', '鲁', '蒙', '闽', '宁', '青', '琼',
           '陕', '苏', '晋', '皖', '湘', '新', '豫', '渝', '粤', '云',
           '藏', '浙']


class char_cnn_net:
    def __init__(self):
        self.dataset = numbers + alphbets + chinese
        self.dataset_len = len(self.dataset)
        self.img_size = 20
        self.y_size = len(self.dataset)
        self.batch_size = 100

        self.x_place = tf.placeholder(dtype=tf.float32, shape=[None, self.img_size, self.img_size], name='x_place')
        self.y_place = tf.placeholder(dtype=tf.float32, shape=[None, self.y_size], name='y_place')
        self.keep_place = tf.placeholder(dtype=tf.float32, name='keep_place')

    def hist_image(self, img):
        assert img.ndim == 2
        hist = [0 for i in range(256)]
        img_h, img_w = img.shape[0], img.shape[1]

        for row in range(img_h):
            for col in range(img_w):
                hist[img[row, col]] += 1
        p = [hist[n] / (img_w * img_h) for n in range(256)]
        p1 = np.cumsum(p)
        for row in range(img_h):
            for col in range(img_w):
                v = img[row, col]
                img[row, col] = p1[v] * 255
        return img

    def find_board_area(self, img):
        assert img.ndim == 2
        img_h, img_w = img.shape[0], img.shape[1]
        top, bottom, left, right = 0, img_h, 0, img_w
        flag = False
        h_proj = [0 for i in range(img_h)]
        v_proj = [0 for i in range(img_w)]

        for row in range(round(img_h * 0.5), round(img_h * 0.8), 3):
            for col in range(img_w):
                if img[row, col] == 255:
                    h_proj[row] += 1
            if flag == False and h_proj[row] > 12:
                flag = True
                top = row
            if flag == True and row > top + 8 and h_proj[row] < 12:
                bottom = row
                flag = False

        for col in range(round(img_w * 0.3), img_w, 1):
            for row in range(top, bottom, 1):
                if img[row, col] == 255:
                    v_proj[col] += 1
            if flag == False and (v_proj[col] > 10 or v_proj[col] - v_proj[col - 1] > 5):
                left = col
                break
        return left, top, 120, bottom - top - 10

    def verify_scale(self, rotate_rect):
        error = 0.4
        aspect = 4  # 4.7272
        min_area = 10 * (10 * aspect)
        max_area = 150 * (150 * aspect)
        min_aspect = aspect * (1 - error)
        max_aspect = aspect * (1 + error)
        theta = 30

        # 宽或高为0，不满足矩形直接返回False
        if rotate_rect[1][0] == 0 or rotate_rect[1][1] == 0:
            return False

        r = rotate_rect[1][0] / rotate_rect[1][1]
        r = max(r, 1 / r)
        area = rotate_rect[1][0] * rotate_rect[1][1]
        if area > min_area and area < max_area and r > min_aspect and r < max_aspect:
            # 矩形的倾斜角度在不超过theta
            if ((rotate_rect[1][0] < rotate_rect[1][1] and rotate_rect[2] >= -90 and rotate_rect[2] < -(90 - theta)) or
                    (rotate_rect[1][1] < rotate_rect[1][0] and rotate_rect[2] > -theta and rotate_rect[2] <= 0)):
                return True
        return False

    def img_Transform(self, car_rect, image):
        img_h, img_w = image.shape[:2]
        rect_w, rect_h = car_rect[1][0], car_rect[1][1]
        angle = car_rect[2]

        return_flag = False
        if car_rect[2] == 0:
            return_flag = True
        if car_rect[2] == -90 and rect_w < rect_h:
            rect_w, rect_h = rect_h, rect_w
            return_flag = True
        if return_flag:
            car_img = image[int(car_rect[0][1] - rect_h / 2):int(car_rect[0][1] + rect_h / 2),
                      int(car_rect[0][0] - rect_w / 2):int(car_rect[0][0] + rect_w / 2)]
            return car_img

        car_rect = (car_rect[0], (rect_w, rect_h), angle)
        box = cv2.boxPoints(car_rect)

        heigth_point = right_point = [0, 0]
        left_point = low_point = [car_rect[0][0], car_rect[0][1]]
        for point in box:
            if left_point[0] > point[0]:
                left_point = point
            if low_point[1] > point[1]:
                low_point = point
            if heigth_point[1] < point[1]:
                heigth_point = point
            if right_point[0] < point[0]:
                right_point = point

        if left_point[1] <= right_point[1]:  # 正角度
            new_right_point = [right_point[0], heigth_point[1]]
            pts1 = np.float32([left_point, heigth_point, right_point])
            pts2 = np.float32([left_point, heigth_point, new_right_point])  # 字符只是高度需要改变
            M = cv2.getAffineTransform(pts1, pts2)
            dst = cv2.warpAffine(image, M, (round(img_w * 2), round(img_h * 2)))
            car_img = dst[int(left_point[1]):int(heigth_point[1]), int(left_point[0]):int(new_right_point[0])]

        elif left_point[1] > right_point[1]:  # 负角度
            new_left_point = [left_point[0], heigth_point[1]]
            pts1 = np.float32([left_point, heigth_point, right_point])
            pts2 = np.float32([new_left_point, heigth_point, right_point])  # 字符只是高度需要改变
            M = cv2.getAffineTransform(pts1, pts2)
            dst = cv2.warpAffine(image, M, (round(img_w * 2), round(img_h * 2)))
            car_img = dst[int(right_point[1]):int(heigth_point[1]), int(new_left_point[0]):int(right_point[0])]

        return car_img

    def pre_process(self, orig_img):
        gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('gray_img', gray_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        blur_img = cv2.blur(gray_img, (3, 3))
        cv2.imshow('blur', blur_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        sobel_img = cv2.Sobel(blur_img, cv2.CV_16S, 1, 0, ksize=3)
        sobel_img = cv2.convertScaleAbs(sobel_img)
        cv2.imshow('sobel', sobel_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        hsv_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)

        h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
        # 黄色色调区间[26，34],蓝色色调区间:[100,124]
        blue_img = (((h > 26) & (h < 34)) | ((h > 100) & (h < 124))) & (s > 70) & (v > 70)
        blue_img = blue_img.astype('float32')

        mix_img = np.multiply(sobel_img, blue_img)
        cv2.imshow('mix', mix_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        mix_img = mix_img.astype(np.uint8)

        ret, binary_img = cv2.threshold(mix_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cv2.imshow('binary',binary_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
        close_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('close', close_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return close_img

    # 给候选车牌区域做漫水填充算法，一方面补全上一步求轮廓可能存在轮廓歪曲的问题，
    # 另一方面也可以将非车牌区排除掉
    def verify_color(self, rotate_rect, src_image):
        img_h, img_w = src_image.shape[:2]
        mask = np.zeros(shape=[img_h + 2, img_w + 2], dtype=np.uint8)
        connectivity = 4  # 种子点上下左右4邻域与种子颜色值在[loDiff,upDiff]的被涂成new_value，也可设置8邻域
        loDiff, upDiff = 30, 30
        new_value = 255
        flags = connectivity
        flags |= cv2.FLOODFILL_FIXED_RANGE  # 考虑当前像素与种子象素之间的差，不设置的话则和邻域像素比较
        flags |= new_value << 8
        flags |= cv2.FLOODFILL_MASK_ONLY  # 设置这个标识符则不会去填充改变原始图像，而是去填充掩模图像（mask）

        rand_seed_num = 5000  # 生成多个随机种子
        valid_seed_num = 200  # 从rand_seed_num中随机挑选valid_seed_num个有效种子
        adjust_param = 0.1
        box_points = cv2.boxPoints(rotate_rect)
        box_points_x = [n[0] for n in box_points]
        box_points_x.sort(reverse=False)
        adjust_x = int((box_points_x[2] - box_points_x[1]) * adjust_param)
        col_range = [box_points_x[1] + adjust_x, box_points_x[2] - adjust_x]
        box_points_y = [n[1] for n in box_points]
        box_points_y.sort(reverse=False)
        adjust_y = int((box_points_y[2] - box_points_y[1]) * adjust_param)
        row_range = [box_points_y[1] + adjust_y, box_points_y[2] - adjust_y]
        # 如果以上方法种子点在水平或垂直方向可移动的范围很小，则采用旋转矩阵对角线来设置随机种子点
        if (col_range[1] - col_range[0]) / (box_points_x[3] - box_points_x[0]) < 0.4 \
                or (row_range[1] - row_range[0]) / (box_points_y[3] - box_points_y[0]) < 0.4:
            points_row = []
            points_col = []
            for i in range(2):
                pt1, pt2 = box_points[i], box_points[i + 2]
                x_adjust, y_adjust = int(adjust_param * (abs(pt1[0] - pt2[0]))), int(
                    adjust_param * (abs(pt1[1] - pt2[1])))
                if (pt1[0] <= pt2[0]):
                    pt1[0], pt2[0] = pt1[0] + x_adjust, pt2[0] - x_adjust
                else:
                    pt1[0], pt2[0] = pt1[0] - x_adjust, pt2[0] + x_adjust
                if (pt1[1] <= pt2[1]):
                    pt1[1], pt2[1] = pt1[1] + adjust_y, pt2[1] - adjust_y
                else:
                    pt1[1], pt2[1] = pt1[1] - y_adjust, pt2[1] + y_adjust
                temp_list_x = [int(x) for x in np.linspace(pt1[0], pt2[0], int(rand_seed_num / 2))]
                temp_list_y = [int(y) for y in np.linspace(pt1[1], pt2[1], int(rand_seed_num / 2))]
                points_col.extend(temp_list_x)
                points_row.extend(temp_list_y)
        else:
            points_row = np.random.randint(row_range[0], row_range[1], size=rand_seed_num)
            points_col = np.linspace(col_range[0], col_range[1], num=rand_seed_num).astype(np.int)

        points_row = np.array(points_row)
        points_col = np.array(points_col)
        hsv_img = cv2.cvtColor(src_image, cv2.COLOR_BGR2HSV)
        h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
        # 将随机生成的多个种子依次做漫水填充,理想情况是整个车牌被填充
        flood_img = src_image.copy()
        seed_cnt = 0
        for i in range(rand_seed_num):
            rand_index = np.random.choice(rand_seed_num, 1, replace=False)
            row, col = points_row[rand_index], points_col[rand_index]
            # 限制随机种子必须是车牌背景色
            if (((h[row, col] > 26) & (h[row, col] < 34)) | ((h[row, col] > 100) & (h[row, col] < 124))) & (
                        s[row, col] > 70) & (v[row, col] > 70):
                cv2.floodFill(src_image, mask, (col, row), (255, 255, 255), (loDiff,) * 3, (upDiff,) * 3, flags)
                cv2.circle(flood_img, center=(col, row), radius=2, color=(0, 0, 255), thickness=2)
                seed_cnt += 1
                if seed_cnt >= valid_seed_num:
                    break
        # ======================调试用======================#
        # show_seed = np.random.uniform(1, 100, 1).astype(np.uint16)
        # cv2.imshow('floodfill' + str(show_seed), flood_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imshow('flood_mask' + str(show_seed), mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # ======================调试用======================#
        # 获取掩模上被填充点的像素点，并求点集的最小外接矩形
        mask_points = []
        for row in range(1, img_h + 1):
            for col in range(1, img_w + 1):
                if mask[row, col] != 0:
                    mask_points.append((col - 1, row - 1))
        mask_rotateRect = cv2.minAreaRect(np.array(mask_points))
        if self.verify_scale(mask_rotateRect):
            return True, mask_rotateRect
        else:
            return False, mask_rotateRect

    # 车牌定位
    def locate_carPlate(self, orig_img, pred_image):
        carPlate_list = []
        temp1_orig_img = orig_img.copy()  # 调试用
        temp2_orig_img = orig_img.copy()  # 调试用
        cloneImg, contours, heriachy = cv2.findContours(pred_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i, contour in enumerate(contours):
            cv2.drawContours(temp1_orig_img, contours, i, (0, 255, 255), 2)
            # 获取轮廓最小外接矩形，返回值rotate_rect
            rotate_rect = cv2.minAreaRect(contour)
            # 根据矩形面积大小和长宽比判断是否是车牌
            if self.verify_scale(rotate_rect):
                ret, rotate_rect2 = self.verify_color(rotate_rect, temp2_orig_img)
                if ret == False:
                    continue
                # 车牌位置矫正
                car_plate = self.img_Transform(rotate_rect2, temp2_orig_img)
                car_plate = cv2.resize(car_plate, (car_plate_w, car_plate_h))  # 调整尺寸为后面CNN车牌识别做准备
                cv2.imshow('car_plate', car_plate)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                # ========================调试看效果========================#
                box = cv2.boxPoints(rotate_rect2)
                for k in range(4):
                    n1, n2 = k % 4, (k + 1) % 4
                    cv2.line(temp1_orig_img, (box[n1][0], box[n1][1]), (box[n2][0], box[n2][1]), (255, 0, 0), 2)
                cv2.imwrite(os.path.join(car_plate_dir, 'plate_down.jpg'), car_plate)
        return carPlate_list

    # 左右切割
    def horizontal_cut_chars(self, plate):
        char_addr_list = []
        area_left, area_right, char_left, char_right = 0, 0, 0, 0
        img_w = plate.shape[1]

        # 获取车牌每列边缘像素点个数
        def getColSum(img, col):
            sum = 0
            for i in range(img.shape[0]):
                sum += round(img[i, col] / 255)
            return sum

        sum = 0
        for col in range(img_w):
            sum += getColSum(plate, col)
        # 每列边缘像素点必须超过均值的60%才能判断属于字符区域
        col_limit = 0  # round(0.5*sum/img_w)
        # 每个字符宽度也进行限制
        charWid_limit = [round(img_w / 12), round(img_w / 5)]
        is_char_flag = False

        for i in range(img_w):
            colValue = getColSum(plate, i)
            if colValue > col_limit:
                if is_char_flag == False:
                    area_right = round((i + char_right) / 2)
                    area_width = area_right - area_left
                    char_width = char_right - char_left
                    if (area_width > charWid_limit[0]) and (area_width < charWid_limit[1]):
                        char_addr_list.append((area_left, area_right, char_width))
                    char_left = i
                    area_left = round((char_left + char_right) / 2)
                    is_char_flag = True
            else:
                if is_char_flag == True:
                    char_right = i - 1
                    is_char_flag = False
        # 手动结束最后未完成的字符分割
        if area_right < char_left:
            area_right, char_right = img_w, img_w
            area_width = area_right - area_left
            char_width = char_right - char_left
            if (area_width > charWid_limit[0]) and (area_width < charWid_limit[1]):
                char_addr_list.append((area_left, area_right, char_width))
        return char_addr_list

    def get_chars(self, car_plate):
        img_h, img_w = car_plate.shape[:2]
        h_proj_list = []  # 水平投影长度列表
        h_temp_len, v_temp_len = 0, 0
        h_startIndex, h_end_index = 0, 0  # 水平投影记索引
        h_proj_limit = [0.2, 0.8]  # 车牌在水平方向得轮廓长度少于20%或多余80%过滤掉
        char_imgs = []

        # 将二值化的车牌水平投影到Y轴，计算投影后的连续长度，连续投影长度可能不止一段
        h_count = [0 for i in range(img_h)]
        for row in range(img_h):
            temp_cnt = 0
            for col in range(img_w):
                if car_plate[row, col] == 255:
                    temp_cnt += 1
            h_count[row] = temp_cnt
            if temp_cnt / img_w < h_proj_limit[0] or temp_cnt / img_w > h_proj_limit[1]:
                if h_temp_len != 0:
                    h_end_index = row - 1
                    h_proj_list.append((h_startIndex, h_end_index))
                    h_temp_len = 0
                continue
            if temp_cnt > 0:
                if h_temp_len == 0:
                    h_startIndex = row
                    h_temp_len = 1
                else:
                    h_temp_len += 1
            else:
                if h_temp_len > 0:
                    h_end_index = row - 1
                    h_proj_list.append((h_startIndex, h_end_index))
                    h_temp_len = 0

        # 手动结束最后得水平投影长度累加
        if h_temp_len != 0:
            h_end_index = img_h - 1
            h_proj_list.append((h_startIndex, h_end_index))
        # 选出最长的投影，该投影长度占整个截取车牌高度的比值必须大于0.5
        h_maxIndex, h_maxHeight = 0, 0
        for i, (start, end) in enumerate(h_proj_list):
            if h_maxHeight < (end - start):
                h_maxHeight = (end - start)
                h_maxIndex = i
        if h_maxHeight / img_h < 0.5:
            return char_imgs
        chars_top, chars_bottom = h_proj_list[h_maxIndex][0], h_proj_list[h_maxIndex][1]

        plates = car_plate[chars_top:chars_bottom + 1, :]
        cv2.imshow('car_plate', car_plate)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow('plate', plates)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite(os.path.join(base_dir, 'plate/car.jpg'), car_plate)
        # cv2.imwrite('/home/yangguang/Projects/CarPlateIdentity-master/plate/car.jpg', car_plate)
        cv2.imwrite(os.path.join(base_dir, 'plate/plates.jpg'), plates)
        # cv2.imwrite('/home/yangguang/Projects/CarPlateIdentity-master/plate/plate.jpg', plates)
        char_addr_list = self.horizontal_cut_chars(plates)

        for i, addr in enumerate(char_addr_list):
            char_img = car_plate[chars_top:chars_bottom + 1, addr[0]:addr[1]]
            char_img = cv2.resize(char_img, (char_w, char_h))
            cv2.imwrite(os.path.join(base_dir, 'cut', str(i) + '.jpg') , char_img)

        return char_imgs


    def extract_char(self, car_plate):
        gray_plate = cv2.cvtColor(car_plate, cv2.COLOR_BGR2GRAY)
        ret, binary_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        char_img_list = self.get_chars(binary_plate)
        # return char_img_list

    # 创建网络
    def cnn_construct(self):
        # 将输入reshape成 20 * 20 * 1
        x_input = tf.reshape(self.x_place, shape=[-1, 20, 20, 1])

        # （卷积层 + 池化层 + dropout层）* 1
        cw1 = tf.Variable(tf.random_normal(shape=[3, 3, 1, 32], stddev=0.01), dtype=tf.float32)
        cb1 = tf.Variable(tf.random_normal(shape=[32]), dtype=tf.float32)
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x_input, filter=cw1, strides=[1, 1, 1, 1], padding='SAME'), cb1))
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv1 = tf.nn.dropout(conv1, self.keep_place)

        # （卷积层 + 池化层 + dropout层）* 2
        cw2 = tf.Variable(tf.random_normal(shape=[3, 3, 32, 64], stddev=0.01), dtype=tf.float32)
        cb2 = tf.Variable(tf.random_normal(shape=[64]), dtype=tf.float32)
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, filter=cw2, strides=[1, 1, 1, 1], padding='SAME'), cb2))
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.dropout(conv2, self.keep_place)

        # （卷积层 + 池化层 + dropout层）* 3
        cw3 = tf.Variable(tf.random_normal(shape=[3, 3, 64, 128], stddev=0.01), dtype=tf.float32)
        cb3 = tf.Variable(tf.random_normal(shape=[128]), dtype=tf.float32)
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, filter=cw3, strides=[1, 1, 1, 1], padding='SAME'), cb3))
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv3 = tf.nn.dropout(conv3, self.keep_place)

        # 将输出reshape成 3 * 3 * 128
        conv_out = tf.reshape(conv3, shape=[-1, 3 * 3 * 128])

        # 3层全连接层
        # 第一层 node = 1024
        fw1 = tf.Variable(tf.random_normal(shape=[3 * 3 * 128, 1024], stddev=0.01), dtype=tf.float32)
        fb1 = tf.Variable(tf.random_normal(shape=[1024]), dtype=tf.float32)
        fully1 = tf.nn.relu(tf.add(tf.matmul(conv_out, fw1), fb1))
        fully1 = tf.nn.dropout(fully1, self.keep_place)

        # 第二层 node = 1024
        fw2 = tf.Variable(tf.random_normal(shape=[1024, 1024], stddev=0.01), dtype=tf.float32)
        fb2 = tf.Variable(tf.random_normal(shape=[1024]), dtype=tf.float32)
        fully2 = tf.nn.relu(tf.add(tf.matmul(fully1, fw2), fb2))
        fully2 = tf.nn.dropout(fully2, self.keep_place)

        # 第三层 node = 类别数
        fw3 = tf.Variable(tf.random_normal(shape=[1024, self.dataset_len], stddev=0.01), dtype=tf.float32)
        fb3 = tf.Variable(tf.random_normal(shape=[self.dataset_len]), dtype=tf.float32)
        fully3 = tf.add(tf.matmul(fully2, fw3), fb3, name='out_put')

        # 输出预测值
        return fully3

    # 训练模型
    def train(self, data_dir, save_model_path):
        print('读取训练数据集：')
        X, y = self.init_data(data_dir)
        print('成功读取 ' + str(len(y)) + ' 个数据')

        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)

        # CNN最后一个全连接层的输出
        out_put = self.cnn_construct()
        # 使用softmax得到预测分数，并找到预测分数最高的标签
        predicts = tf.nn.softmax(out_put)
        predicts = tf.argmax(predicts, axis=1)
        # 得到原始样本真正的标签
        actual_y = tf.argmax(self.y_place, axis=1)
        # 求准确率
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predicts, actual_y), dtype=tf.float32))
        # 设置损失函数，并设置优化器
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_put, labels=self.y_place))
        opt = tf.train.AdamOptimizer(learning_rate=0.001)
        # 最小化损失函数
        train_step = opt.minimize(cost)


        # 开始绘画
        with tf.Session() as sess:
            # 初始化参数
            init = tf.global_variables_initializer()
            sess.run(init)
            step = 0
            saver = tf.train.Saver()
            # 循环
            while True:
                train_index = np.random.choice(len(train_x), self.batch_size, replace=False)
                train_randx = train_x[train_index]
                train_randy = train_y[train_index]
                _, loss = sess.run([train_step, cost],
                                   feed_dict={self.x_place: train_randx, self.y_place: train_randy,
                                              self.keep_place: 0.75})
                step += 1

                if step % 10 == 0:
                    test_index = np.random.choice(len(test_x), self.batch_size, replace=False)
                    test_randx = test_x[test_index]
                    test_randy = test_y[test_index]
                    acc = sess.run(accuracy, feed_dict={self.x_place: test_randx, self.y_place: test_randy,
                                                        self.keep_place: 1.0})
                    print(step, loss)
                    if step % 50 == 0:
                        print('accuracy:' + str(acc))
                    if step % 500 == 0:
                        saver.save(sess, save_model_path, global_step=step)
                        break
                    # if acc > 0.99 and step > 500:
                    #     saver.save(sess, save_model_path, global_step=step)
                    #     break

    def test(self, x_images, model_path):
        text_list = []
        out_put = self.cnn_construct()
        predicts = tf.nn.softmax(out_put)
        predicts = tf.argmax(predicts, axis=1)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, model_path)
            preds = sess.run(predicts, feed_dict={self.x_place: x_images, self.keep_place: 1.0})
            for i in range(len(preds)):
                pred = preds[i].astype(int)
                text_list.append(self.dataset[pred])
            return text_list

    def list_all_files(self, root):
        files = []
        list = os.listdir(root)
        for i in range(len(list)):
            element = os.path.join(root, list[i])
            if os.path.isdir(element):
                temp_dir = os.path.split(element)[-1]
                if temp_dir in self.dataset:
                    files.extend(self.list_all_files(element))
            elif os.path.isfile(element):
                files.append(element)
        return files

    def init_testData(self, dir):
        test_X = []
        if not os.path.exists(test_dir):
            raise ValueError('没有找到文件夹')
        for i in range(0, 7):
            fname = os.path.join(dir, str(i) + '.jpg')
            src_img = cv2.imread(fname, cv2.COLOR_BGR2GRAY)
            if src_img.ndim == 3:
                continue
            resize_img = cv2.resize(src_img, (20, 20))
            test_X.append(resize_img)
        test_X = np.array(test_X)
        return test_X

    def init_data(self, dir):
        X = []
        y = []
        if not os.path.exists(data_dir):
            raise ValueError('没有找到文件夹')
        files = self.list_all_files(dir)

        for file in files:
            src_img = cv2.imread(file, cv2.COLOR_BGR2GRAY)
            if src_img.ndim == 3:
                continue
            resize_img = cv2.resize(src_img, (20, 20))
            X.append(resize_img)
            # 获取图片文件全目录
            dir = os.path.dirname(file)
            # 获取图片文件上一级目录名
            dir_name = os.path.split(dir)[-1]
            vector_y = [0 for i in range(len(self.dataset))]
            index_y = self.dataset.index(dir_name)
            vector_y[index_y] = 1
            y.append(vector_y)

        X = np.array(X)
        y = np.array(y).reshape(-1, self.dataset_len)
        return X, y

if __name__ == '__main__':

    # 训练模型设为1，测试模型设为0
    train_flag = 0

    # 车牌宽高
    car_plate_w, car_plate_h = 136, 36

    # 字符宽高
    char_w, char_h = 20, 20

    base_dir = 'C:/Users/yangguang/PycharmProjects/CarPlateIdentity-master'

    # 训练数据集
    data_dir = os.path.join(base_dir, 'images/cnn_char_train')
    # 测试数据集
    test_dir = os.path.join(base_dir, 'cut')
    # 模型保存路径
    train_model_path = os.path.join(base_dir, 'model/char/model.ckpt')
    # 模型读取路径
    model_path = os.path.join(base_dir, 'model/char/model.ckpt-520')

    # 读取自然车牌图片
    img = cv2.imread(os.path.join(base_dir, 'images/pictures/10.jpg'))
    # 载入class
    net = char_cnn_net()

    # 预处理图片找到车牌位置'
    pred_img = net.pre_process(img)

    # 车牌保存路径
    car_plate_dir = os.path.join(base_dir, 'plate')

    # 车牌定位
    car_plate_list = net.locate_carPlate(img, pred_img)

    # 读取处理后的车牌
    car_plate = cv2.imread(os.path.join(car_plate_dir, 'plate_down.jpg'))

    # 字符分割
    net.extract_char(car_plate)

    if train_flag == 1:
        # 训练模型
        net.train(data_dir, train_model_path)
    else:
        # 测试部分
        test_X = net.init_testData(test_dir)
        text = net.test(test_X, model_path)
        print('车牌号为 ' + str(text))
