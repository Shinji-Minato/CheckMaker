#システムのbin中のpythonを使用する
import numpy as np
from sklearn import datasets
from collections import Counter
from PIL import Image, ImageFilter, ImageDraw, ImageFont

#行列表示
np.set_printoptions(threshold=np.inf)

SIZE_OF_RESIZE =50
ROW = 3; #将棋盤の行数
COLUMN =3; #将棋盤の列数


#駒データ読み込み
koma =[]

#0:王(自), 1:王(敵), 2:歩兵(自), 3:歩兵(敵), 4:銀(自), 5:銀(敵), 6:金(自), 7:金(敵), 8:駒なし
img = Image.open('/home/minato/OneDrive/Ubuntu/image/ou.png')
koma.append(img)
koma[0] = koma[0].convert('L')
koma[0] = koma[0].resize((SIZE_OF_RESIZE, SIZE_OF_RESIZE))
koma.append(img)
koma[1] = koma[1].transpose(Image.FLIP_LEFT_RIGHT)
koma[1] = koma[1].transpose(Image.FLIP_TOP_BOTTOM)
koma[1] = koma[1].convert('L')
koma[1] = koma[1].resize((SIZE_OF_RESIZE, SIZE_OF_RESIZE))

img = Image.open('/home/minato/OneDrive/Ubuntu/image/fu.png')
koma.append(img)
koma[2] = koma[2].convert('L')
koma[2] = koma[2].resize((SIZE_OF_RESIZE, SIZE_OF_RESIZE))
koma.append(img)
koma[3] = koma[3].transpose(Image.FLIP_LEFT_RIGHT)
koma[3] = koma[3].transpose(Image.FLIP_TOP_BOTTOM)
koma[3] = koma[3].convert('L')
koma[3] = koma[3].resize((SIZE_OF_RESIZE, SIZE_OF_RESIZE))


img = Image.open('/home/minato/OneDrive/Ubuntu/image/gin.png')
koma.append(img)
koma[4] = koma[4].convert('L')
koma[4] = koma[4].resize((SIZE_OF_RESIZE, SIZE_OF_RESIZE))
koma.append(img)
koma[5] = koma[5].transpose(Image.FLIP_LEFT_RIGHT)
koma[5] = koma[5].transpose(Image.FLIP_TOP_BOTTOM)
koma[5] = koma[5].convert('L')
koma[5] = koma[5].resize((SIZE_OF_RESIZE, SIZE_OF_RESIZE))

img = Image.open('/home/minato/OneDrive/Ubuntu/image/kin.png')
koma.append(img)
koma[6] = koma[6].convert('L')
koma[6] = koma[6].resize((SIZE_OF_RESIZE, SIZE_OF_RESIZE))
koma.append(img)
koma[7] = koma[7].transpose(Image.FLIP_LEFT_RIGHT)
koma[7] = koma[7].transpose(Image.FLIP_TOP_BOTTOM)
koma[7] = koma[7].convert('L')
koma[7] = koma[7].resize((SIZE_OF_RESIZE, SIZE_OF_RESIZE))

img = Image.open('/home/minato/OneDrive/Ubuntu/image/ban_original.png')
koma.append(img)
koma[8] = koma[8].convert('L')
koma[8] = koma[8].resize((SIZE_OF_RESIZE, SIZE_OF_RESIZE))

arr = np.empty((0, SIZE_OF_RESIZE**2+1), int)

for i in range(9):
    koma_array = []
    for y in range(SIZE_OF_RESIZE):
        for x in range(SIZE_OF_RESIZE):
            koma_array.append(koma[i].getpixel((x, y)))
    koma_array.append(i)
    arr = np.append(arr, np.array([koma_array]), axis=0)


koma[0].save('/home/minato/OneDrive/Ubuntu/image/0.png', quality=95)
koma[1].save('/home/minato/OneDrive/Ubuntu/image/1.png', quality=95)
koma[2].save('/home/minato/OneDrive/Ubuntu/image/2.png', quality=95)
koma[3].save('/home/minato/OneDrive/Ubuntu/image/3.png', quality=95)
koma[4].save('/home/minato/OneDrive/Ubuntu/image/4.png', quality=95)
koma[5].save('/home/minato/OneDrive/Ubuntu/image/5.png', quality=95)
koma[6].save('/home/minato/OneDrive/Ubuntu/image/6.png', quality=95)
koma[7].save('/home/minato/OneDrive/Ubuntu/image/7.png', quality=95)
koma[8].save('/home/minato/OneDrive/Ubuntu/image/8.png', quality=95)

#将棋盤データ読み込み

test_data_original = []
test_data_resized = []
ban = Image.open('/home/minato/OneDrive/Ubuntu/image/banmen1.png').convert('RGB')

size_of_x, size_of_y = ban.size
size_of_x = int(size_of_x/COLUMN) #グリッド一つのサイズ
size_of_y = int(size_of_y/ROW)

for y in range(ROW): #将棋盤から画像の切り出し
    for x in range(COLUMN):
        test_data_original.append(ban)
        test_data_original[x + y*COLUMN] = test_data_original[x + y*COLUMN].crop((size_of_x*x+5, size_of_y*y+5, size_of_x*(x+1)-5, size_of_y*(y+1)-5))

        test_data_resized.append(test_data_original[x + y*COLUMN])
        test_data_resized[x + y*COLUMN] = test_data_resized[x + y*COLUMN].convert('L')
        test_data_resized[x + y*COLUMN] = test_data_resized[x + y*COLUMN].resize((SIZE_OF_RESIZE, SIZE_OF_RESIZE))

arr_ban = np.empty((0, SIZE_OF_RESIZE**2), int)

for i in range(ROW*COLUMN):
    ban_array = []
    for y in range(SIZE_OF_RESIZE):
        for x in range(SIZE_OF_RESIZE):
            ban_array.append(test_data_resized[i].getpixel((x, y)))
    arr_ban = np.append(arr_ban, np.array([ban_array]), axis=0)


test_data_resized[0].save('/home/minato/OneDrive/Ubuntu/image/ban0.png', quality=95)
test_data_resized[1].save('/home/minato/OneDrive/Ubuntu/image/ban1.png', quality=95)
test_data_resized[2].save('/home/minato/OneDrive/Ubuntu/image/ban2.png', quality=95)
test_data_resized[3].save('/home/minato/OneDrive/Ubuntu/image/ban3.png', quality=95)
test_data_resized[4].save('/home/minato/OneDrive/Ubuntu/image/ban4.png', quality=95)
test_data_resized[5].save('/home/minato/OneDrive/Ubuntu/image/ban5.png', quality=95)
test_data_resized[6].save('/home/minato/OneDrive/Ubuntu/image/ban6.png', quality=95)
test_data_resized[7].save('/home/minato/OneDrive/Ubuntu/image/ban7.png', quality=95)
test_data_resized[8].save('/home/minato/OneDrive/Ubuntu/image/ban8.png', quality=95)


#k近傍法

train_data = arr
test_data = arr_ban

def knn(k, train_data, test_data):
    labels = []

    for test in test_data:
        # 1. すべてのトレインデータとtest（このループステップでラベルを予測したいデータ）との距離を計算したリストを作る
        distances = np.sum((train_data[:, :-1] - test[:]) ** 2, axis=1)

        # 2. 距離リストの値が小さい順に並べた、トレインデータのインデックスを持つリストを作る
        sorted_train_indexes = np.argsort(distances)

        # 3. インデックスリストを元に、testから近いk個のトレインデータのラベルを取り出す
        sorted_k_labels = train_data[sorted_train_indexes, -1][:k]

        # 4. sorted_k_labelsの中で最も数の多かったlabelを取り出す
        label = Counter(sorted_k_labels).most_common(1)[0][0]
        labels.append(label)
    return labels


pred_labels = knn(1, train_data, test_data)

print(pred_labels)

#反転する駒の判定

#xの要素がリストの何番目かを返す関数
#def my_index_multi(l, x):
#    return [i for i, _x in enumerate(l) if _x == x]

#print(my_index_multi(pred_labels, 0))
#pos_of_ou = my_index_multi(pred_labels, 0)

pos_of_ou = pred_labels.index(0)
ou_y = int(pos_of_ou / COLUMN)
ou_x = pos_of_ou - ou_y*COLUMN

reverse_x = []
reverse_y = []

if ((ou_x-1)>=0) and ((ou_y-1)>=0):
    if pred_labels [(ou_y-1)*COLUMN+(ou_x-1)] in {4,6}:
        reverse_x.append(ou_x-1)
        reverse_y.append(ou_y-1)

if ((ou_x)>=0) and ((ou_y-1)>=0):
    if pred_labels [(ou_y-1)*COLUMN+(ou_x)] in {2,4,6}:
        reverse_x.append(ou_x)
        reverse_y.append(ou_y-1)

if ((ou_x+1)<COLUMN) and ((ou_y-1)>=0):
     if pred_labels [(ou_y-1)*COLUMN+(ou_x+1)] in {4,6}:
        reverse_x.append(ou_x+1)
        reverse_y.append(ou_y-1)

if ((ou_x-1)>=0) and ((ou_y)>=0):
     if pred_labels [(ou_y)*COLUMN+(ou_x-1)] in {6}:
        reverse_x.append(ou_x-1)
        reverse_y.append(ou_y)

if ((ou_x+1)<COLUMN) and ((ou_y)>=0):
     if pred_labels [(ou_y)*COLUMN+(ou_x+1)] in {6}:
        reverse_x.append(ou_x+1)
        reverse_y.append(ou_y)

if ((ou_x-1) >= 0) and ((ou_y+1) < ROW):
    if pred_labels[(ou_y + 1) * COLUMN + (ou_x - 1)] in {4}:
        reverse_x.append(ou_x - 1)
        reverse_y.append(ou_y + 1)

if ((ou_x) >= 0) and ((ou_y+1) < ROW):
    if pred_labels[(ou_y + 1) * COLUMN + (ou_x)] in {4, 6}:
        reverse_x.append(ou_x)
        reverse_y.append(ou_y + 1)

if ((ou_x+1) < COLUMN) and ((ou_y+1) < ROW):
    if pred_labels[(ou_y + 1) * COLUMN + (ou_x + 1)] in {4}:
        reverse_x.append(ou_x + 1)
        reverse_y.append(ou_y + 1)


#画像の生成

if len(reverse_x) == 0 :
    print('王手できる駒がありません')
    thumb = Image.open('/home/minato/OneDrive/Ubuntu/image/no_check.jpg')
    thumb = thumb.resize((600, 160))
    ban.paste(thumb, ((75,295)))

for i in range(len(reverse_x)):
    thumb = test_data_original[reverse_x[i] + reverse_y[i] * COLUMN]
    thumb = thumb.transpose(Image.FLIP_LEFT_RIGHT)
    thumb = thumb.transpose(Image.FLIP_TOP_BOTTOM)
    ban.paste(thumb, (size_of_x * reverse_x[i] + 5, size_of_y * reverse_y[i] + 5))

    draw = ImageDraw.Draw(ban)
    draw.line((size_of_x*reverse_x[i], size_of_y*reverse_y[i]+3, size_of_x*(reverse_x[i]+1), size_of_y*reverse_y[i]+3), fill=(255,0,0), width=8)
    draw.line((size_of_x * (reverse_x[i]+1)-3, size_of_y * reverse_y[i], size_of_x * (reverse_x[i] + 1)-3, size_of_y * (reverse_y[i]+1)), fill=(255, 0, 0), width=8)
    draw.line((size_of_x * reverse_x[i], size_of_y * (reverse_y[i]+1) - 3, size_of_x * (reverse_x[i] + 1),size_of_y * (reverse_y[i]+1) - 3), fill=(255, 0, 0), width=8)
    draw.line((size_of_x * (reverse_x[i])+3, size_of_y * reverse_y[i], size_of_x * (reverse_x[i])+3, size_of_y * (reverse_y[i]+1)), fill=(255, 0, 0), width=8)

ban.save('/home/minato/OneDrive/Ubuntu/image/goal.png', quality=95)
