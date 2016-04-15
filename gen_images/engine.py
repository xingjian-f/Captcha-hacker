#!/usr/bin/env python
# coding=utf-8

from PIL import Image, ImageDraw, ImageFont
import random
import math
import os
import argparse

def operation(x, op, y, mode):
    '''
    return equation string giving x, y, operator, within mode of 'chi' or 'en'
    '''
    lang_mapper = ['零', '壹', '贰', '叁', '肆', '伍', '陆', '柒', '捌', '玖']
    opt_mapper_en = {'+':'加上', '-':'减去', '*':'乘以', '/':'除以'}
    opt_mapper_chi = {'+':'加', '-':'减', '*':'乘'}
    symbol = '？'
    if mode == 'chi':
        x = lang_mapper[x]
        op = opt_mapper_chi[op]
        y = lang_mapper[y]
        symbol = ''
    elif mode == 'en':
        x = str(x)
        op = opt_mapper_en[op]
        y = str(y)
    ret = x + op + y + '等于' + symbol
    return ret.decode('utf-8')

def randomEquation(mode='en'):
    x = random.randint(0, 9)
    y = random.randint(0, 9)
    op = random.choice(['+', '-', '*', '/'])
    if mode == 'chi':           # no '/' for mode = 'chi'
        op = random.choice(['+', '-', '*'])
    # constrains for en
    elif op == '-':             # x is always bigger for x - y for en mode
        if x < y:
            temp = x
            x = y
            y = temp
    elif op == '/':             # to make sure (x % y == 0) and (y != 0)
        y = random.randint(1, 9)
        x = x * y
    equation = operation(x, op, y, mode)
    return equation

ChiSayings = []
def randomChiSaying():
    '''
    generate a random chinese saying from txt
    '''
    if len(ChiSayings) == 0:        # load data first
        filepath = '../trainpic/chisayings.txt'
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = unicode(line.strip(), 'utf-8')
                ChiSayings.append(line)
        print(str(len(ChiSayings)) + ' Chinese Sayings detected!')
    index = random.randint(0, len(ChiSayings)-1)
    return ChiSayings[index]


codec = []
def randomChiChar():
    '''
    generate a random chinese character from txt
    '''
    global codec
    if len(codec) == 0:        # load data first
        filepath = '../trainpic/chisayings.txt'
        with open(filepath, 'r') as f:
            vocab = ''.join([line.strip() for line in f.readlines()])
            codec = list(set(vocab.decode('utf-8')))
        print(str(len(codec)) + ' Hanzi detected!')
    pass
    index = random.randint(0, len(codec)-1)
    return codec[index]


alphabeta = 'ABCDEFGHJKLMNPQRSTUVWXYZ' # no 'I' or 'O'
alphabeta = alphabeta + '123456789' # no '0'

def randomAlphaNum(num = 4):
    num = num or 4
    strings = ''
    for i in range(4):
        index = random.randint(0, len(alphabeta) - 1)
        char = alphabeta[index]
        strings = strings + char
    return strings


class ImageGenerator(object):
    '''
    generate a random image with Class ImageGenerator
    example:
        from generator import ImageGenerator
        ig = ImageGenerator()
        ig.generateImage(string='1776', path='./1.jpg')
    '''
    def __init__(self, fontSize=24, size = (200, 50), bgColor = (200, 200, 200)):
        '''
        declare and initialize private varians
        '''
        self.size = size
        self.bgColor = bgColor
        self.fontSize = fontSize
        self.font = ImageFont.truetype("arial.ttf", 20)
    pass

    def randRGB(self):
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    pass

    def randPoint(self, num=200, color=-1):
        draw = ImageDraw.Draw(self.image)
        for i in range(0, num):
            x = random.randint(0, self.size[0])
            y = random.randint(0, self.size[1])
            if color < 0:
                draw.point((x, y), fill=self.randRGB())
            else:
                draw.point((x, y), fill=(color, color, color))
            pass
        pass
    pass

    def randLine(self, num=30, length=12, color=-1):
        '''
        make some random line noises
        '''
        length = length + random.randint(-length / 4, length / 4)
        draw = ImageDraw.Draw(self.image)
        for i in range(0, num):
            # draw a line from random point(x1, y1) at random angle
            x1 = random.randint(0, self.size[0])
            y1 = random.randint(0, self.size[1])
            angle = random.randint(0, 360) * math.pi / 180
            x2 = x1 + length * math.cos(angle)
            y2 = y1 + length * math.sin(angle)
            if color == -1: # mean randRGB
                draw.line([(x1, y1), (x2, y2)], self.randRGB())
            else:
                draw.line([(x1, y1), (x2, y2)], color)
            pass
        pass
    pass

    def drawChar(self, text, angle=random.randint(-10, 10), color=-1):
        '''
        get a sub image with one specific character
        '''
        charImg = Image.new('RGBA', (int(self.fontSize * 1.3), int(self.fontSize * 1.3)))
        if color == -1:
            ImageDraw.Draw(charImg).text((0, 0), text, font=self.font, fill=self.randRGB())
        else:
            ImageDraw.Draw(charImg).text((0, 0), text, font=self.font, fill=color)
        charImg = charImg.crop(charImg.getbbox())
        charImg = charImg.rotate(angle, Image.BILINEAR, expand=1)
        return charImg
    pass

    def distort(self, dim, A, w, fei):
        fei = random.random() * 2 * math.pi
        retImg = Image.new('RGB', self.size) # black background
        data_ori = list(self.image.getdata())
        data_ret = list(retImg.getdata())
        width, height = self.image.size
        for x in range(width):
            for y in range(height):
                new_x = x + int(A * math.sin(w * y + fei))
                if new_x >= 0 and new_x <= (width - 1):
                    data_ret[y * width + new_x] = data_ori[y * width + x]
        self.image.putdata(data_ret)
    pass

    def italic(self, img, m):
        '''
        shift picture to make italic effect, if m > 0, turn right, else left
        -1 < m < 1
        '''
        width, height = img.size
        xshift = abs(m) * width
        new_width = width + int(round(xshift))
        new_img = img.transform((new_width, height), Image.AFFINE,
                            (1, m, -xshift, 0, 1, 0), Image.BICUBIC)
        return new_img
    pass

    def generateImage(self, strings = u'8除以6等于？', path='out.jpg'):
        '''
        genreate a picture giving string and path
        '''
        self.image = Image.new('RGB', self.size, self.bgColor) # image must be initialized here
        self.randLine()
        gap = 2 # pixes between two characters
        start = random.randint(0, 5)
        for i in range(0, len(strings)):
            charImg = self.drawChar(text=strings[i], angle=random.randint(-60, 60))
            x = start + self.fontSize * i + random.randint(0, gap) * i
            y = (self.image.size[1] - charImg.size[1]) / 2 + random.randint(-10, 10)
            self.image.paste(charImg, (x, y), charImg)
        self.image.save(path)
        return self.image
    pass


class Type2ImageGenerator(ImageGenerator):
    def __init__(self, fontPath, fontSize=26, size = (160, 53), bgColor = (255, 255, 255)):
        super(Type2ImageGenerator, self).__init__(fontPath, fontSize, size, bgColor)
    pass

    def generateImage(self, strings = u'叁加陆等于', path='out.jpg'):
        self.image = Image.new('RGB', self.size, self.bgColor) # image must be initialized here
        self.randPoint()
        self.randLine(num=random.randint(15, 20), length=100, color=-1)
        gap = 5
        start = random.randint(0, 5)
        for i in range(0, len(strings)):
            charImg = self.drawChar(text=strings[i], angle=random.randint(-15, 15))
            x = start + self.fontSize * i + random.randint(1, gap) * i
            y = (self.image.size[1] - charImg.size[1]) / 2 + random.randint(-10, 10)
            self.image.paste(charImg, (x, y), charImg)
        self.image.save(path)
    pass

class Type3ImageGenerator(Type2ImageGenerator):
    def __init__(self, fontPath, fontSize=36):
        super(Type3ImageGenerator, self).__init__(fontPath, fontSize)

    def generateImage(self, strings = u'参差不齐', path='out.jpg'):
        self.image = Image.new('RGB', self.size, self.bgColor) # image must be initialized here
        self.randPoint()
        self.randLine(num=random.randint(15, 20), length=100, color=-1)
        gap = 5
        start = random.randint(0, 5)
        for i in range(0, len(strings)):
            charImg = self.drawChar(text=strings[i], angle=random.randint(-15, 15))
            x = start + self.fontSize * i + random.randint(1, gap) * i
            y = (self.image.size[1] - charImg.size[1]) / 2 + random.randint(-6, 6)
            self.image.paste(charImg, (x, y), charImg)
        self.image.save(path)
    pass

class Type5ImageGenerator(ImageGenerator):
    '''
    share code with type 6 iamge generator
    type 5 looks like type 2
    type 6 looks like type 3
    '''
    def __init__(self, fontPath, fontSize=26, size=(180, 40), bgColor = -1):
        super(Type5ImageGenerator, self).__init__(fontPath, fontSize, size, bgColor)
    pass

    def generateImage(self, strings = u'贰乘叁等于', path='out.jpg'):
        background = self.bgColor
        if self.bgColor == -1:
            background = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
        self.image = Image.new('RGB', self.size, background) # image must be initialized here
        self.randLine(num=random.randint(5, 8), length=100, color=-1)
        self.randLine(num=random.randint(15, 20), length=10, color=(100, 100, 100))
        gap = 1
        start = random.randint(15, 25)
        for i in range(0, len(strings)):
            charcolor = (random.randint(50, 180), random.randint(50, 180), random.randint(50, 180))
            charImg = self.drawChar(text=strings[i], angle=random.randint(-15, 15), color=charcolor)
            x = start + self.fontSize * i + random.randint(1, gap) * i
            y = (self.image.size[1] - charImg.size[1]) / 2 + random.randint(-6, 6)
            self.image.paste(charImg, (x, y), charImg)
        self.image.save(path)
    pass

class Type8ImageGenerator(ImageGenerator):
    def __init__(self, fontPath, fontSize = 22, size=(300, 50), bgColor = 255):
        super(Type8ImageGenerator, self).__init__(fontPath, fontSize, size, bgColor)
    pass

    def generateImage(self, strings = u'陆加上2等于几', path='out.jpg'):
        background = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
        self.image = Image.new('RGB', self.size, background) # image must be initialized here
        lineColor = random.randint(180, 210)
        self.randLine(num=300, length=10, color=(lineColor, lineColor, lineColor))
        gap = 2
        x = random.randint(12, 15)
        for i in range(0, len(strings)):
            charImg = self.drawChar(text=strings[i], angle=0, color=(0, 0, 0))
            charImg = self.italic(charImg, 0.4)
            y = (self.image.size[1] - charImg.size[1]) / 2 - 2
            self.image.paste(charImg, (x, y), charImg)
            x = x + charImg.size[1] - gap
        pass
        self.image.save(path)
    pass


class Type9ImageGenerator(ImageGenerator):
    def __init__(self, fontPath, fontSize = 46, size=(150, 50), bgColor = 255):
        super(Type9ImageGenerator, self).__init__(fontPath, fontSize, size, bgColor)
    pass

    def generateImage(self, strings = u'7URT', path='out.jpg'):
        self.image = Image.new('RGB', self.size, (255, 255, 255)) # image must be initialized here
        self.randLine(num=20, length=15, color=0)
        gap = 5 
        x = random.randint(32, 35)
        for i in range(0, len(strings)):
            charImg = self.drawChar(text=strings[i], angle=0, color=(0, 0, 0))
            y = (self.image.size[1] - charImg.size[1]) / 2 + random.randint(5, 6)
            self.image.paste(charImg, (x, y), charImg)
            x = x + charImg.size[1] - gap
        pass
        self.distort(0, random.randint(3, 8), math.pi / 50, 0)
        self.image.save(path)
    pass

class Type10ImageGenerator(ImageGenerator):
    '''
    single chinese character
    '''
    def __init__(self, fontPath, fontSize=22, size=(22, 40), bgColor = -1):
        super(Type10ImageGenerator, self).__init__(fontPath, fontSize, size, bgColor)
    pass

    def generateImage(self, strings = u'闻', path='out.jpg'):
        old_fontsize = self.fontSize
        self.fontSize = self.fontSize + random.randint(-1, 1)
        self.font = ImageFont.truetype(self.fontPath, self.fontSize)
        self.image = Image.new('RGB', self.size, (255, 255, 255)) # image must be initialized here
        lineColor = random.randint(100, 200)
        self.randLine(num=random.randint(5, 10), length=10, color=(lineColor, lineColor, lineColor))
        charcolor = (random.randint(0, 200), random.randint(0, 200), random.randint(0, 200))
        charImg = self.drawChar(text=strings, angle=0, color=charcolor)
        y = (self.image.size[1] - charImg.size[1]) / 2 + random.randint(-10, 10)
        x = (self.image.size[0] - charImg.size[0]) / 2
        x = x if x > 0 else 0
        self.image.paste(charImg, (x, y), charImg)
        self.image.save(path)
        self.fontSize = old_fontsize
    pass

def syntheticData(args):
    if type(args.fonts) == str:
        args.fonts = [args.fonts]
    if not os.path.isdir(args.savedir):
        os.mkdir(args.savedir)
        print('mkdir ' + args.savedir)
    print('enter ' + args.savedir)
    for font in args.fonts:
        '''
        generate data for every font
        '''
        if args.verbose == True:
            print('reading ' + font)
        ig = []
        if args.type == 1:
            ig = ImageGenerator(font)
        elif args.type == 2:
            ig = Type2ImageGenerator(font)
        elif args.type == 3:
            ig = Type3ImageGenerator(font)
            # no type4 here
        elif args.type == 5 or args.type == 6:
            ig = Type5ImageGenerator(font)
        elif args.type == 8:
            ig = Type8ImageGenerator(font, args.fontsize)
        elif args.type == 9:
            ig = Type9ImageGenerator(font)
        elif args.type == 10:
            ig = Type10ImageGenerator(font, args.fontsize)
        fontname = font.split('.')[-2].split('/')[-1]
        print('generate data for font ' + fontname)
        for i in range(args.number):
            if args.number > 100 and i % (args.number / 10) == 0:
                print i*100/args.number, "%..."
            txtpath = os.path.join(args.savedir, fontname + str(i) + '.gt.txt')
            with open(txtpath, 'w') as f:
                label = ''
                if args.type == 1:
                    label = randomEquation(mode='en')
                elif args.type == 2 or args.type == 5:
                    label = randomEquation(mode='chi')
                elif args.type == 3 or args.type == 6:
                    label = randomChiSaying()
                elif args.type == 8:
                    label = randomEquation()
                elif args.type == 9:
                    label = randomAlphaNum()
                elif args.type == 10:
                    label = randomChiChar()
                f.write(label.encode('utf-8')+ '\n')
                filepath = os.path.join(args.savedir, fontname + str(i) + '.' + args.picformat)
                ig.generateImage(strings=label, path=filepath)
            pass
        pass
    pass

def getAllFonts(fontdir = '../fonts/'):
    fonts = []
    for font in os.listdir(fontdir):
        ext = font.split('.')[1]
        if ext == 'ttf' or ext == 'TTF' or ext == 'ttc':
            fonts.append(os.path.join(fontdir, font))
    print str(len(fonts)) + ' fonts detected!'
    return fonts


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-t", "--type", default=1, type=int, help="which type of CAPTCHA to generate")
    # parser.add_argument("-d", "--savedir", default="../trainpic/temp/", help="directory to save the pictures")
    # parser.add_argument("-f", "--fonts", default=getAllFonts(), help="choose which font to use")
    # parser.add_argument("-fs", "--fontsize", default=24, type=int, help="specify font size")
    # parser.add_argument("-n", "--number", default=2, type=int, help="how many pictures to generate for every font?")
    # parser.add_argument("-p", "--picformat", default="jpg", help="jpg or png? specific filename suffix")

    # feature = parser.add_mutually_exclusive_group(required=False)
    # feature.add_argument("--verbose", dest='verbose', action='store_true', help="print verbose infomation")
    # feature.add_argument("--no-verbose", dest='verbose', action='store_false', help="forbid verbose infomation")
    # parser.set_defaults(feature=True)

    # args = parser.parse_args()
    # #print(args)
    # syntheticData(args)
    a = ImageGenerator()
    im = a.generateImage()
    im.show()