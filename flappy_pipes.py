import pygame as pg
import sys
from pygame.locals import *
from itertools import cycle
import random
from PIL import Image
import os

FPS = 30
SCREENWIDTH = 1280
SCREENHEIGHT = 720
# 上下2棵树间隙高度
TREEGAPSIZE = 180
BASTY = SCREENHEIGHT * 0.863
IMAGES, HITMASKS = {}, {}
DIS = 250

# 10名玩家 （每个玩家有2种飞行状态）
# player 1:down up
# ......
PLAYERS_LIST = []
for i in range(10):
    PLAYERS_LIST.append(('pygame/src/birds_upflap_' + str(i) + '.png',
                         'pygame/src/birds_downflap_' + str(i) + '.png'))
# 2张背景 白天与夜晚
BACKGROUND_LIST = [
    'pygame/src/bg1.png'
    # os.path.join(PROJECT_PATH, 'src/background-night.png')
]
# 上下两棵树
# 上下两棵树
TREE_LIST = ['pygame/src/tree_up.png', 'pygame/src/tree_down.png']


def main():
    global SCREEN, FPSCLOCK
    pg.init()
    # FPSCLOCK = pg.time.Clock()
    SCREEN = pg.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
    pg.display.set_caption('Flappy Bird')
    # ground sprite
    IMAGES['img_ground'] = pg.image.load(
        'pygame/src/ground.png').convert_alpha()

    while True:
        # select random background
        randBg = random.randint(0, len(BACKGROUND_LIST) - 1)
        IMAGES['background'] = pg.image.load(BACKGROUND_LIST[randBg]).convert()

        # select random player
        randPlayer = random.randint(0, len(PLAYERS_LIST) - 1)
        IMAGES['player'] = (pg.image.load(
            PLAYERS_LIST[randPlayer][0]).convert_alpha(),
                            pg.image.load(
                                PLAYERS_LIST[randPlayer][1]).convert_alpha())

        IMAGES['tree'] = (pg.image.load(TREE_LIST[0]).convert_alpha(),
                          pg.image.load(TREE_LIST[1]).convert_alpha())

        HITMASKS['tree'] = (getHitmask(IMAGES['tree'][0]),
                            getHitmask(IMAGES['tree'][1]))
        HITMASKS['player'] = (getHitmask(IMAGES['player'][0]),
                              getHitmask(IMAGES['player'][1]))

        movementInfo = wait()
        crashInfo = game(movementInfo)
        gameOver(crashInfo)


def wait():
    '''show animation before starting'''
    # player blit on screen
    playerIndex = 0
    playerIndexGen = cycle([0, 1])
    # iterator used to change playerIndex after every 5th iteration
    loopIter = 0

    playerx = int(SCREENWIDTH * 0.1)
    playery = int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2)

    basex = 0
    # amount by which base can maximum shift to left
    baseShift = IMAGES['img_ground'].get_width() - SCREENWIDTH

    # player shm for up-down motion on welcome screen
    playerShmVals = {'val': 0, 'dir': 1}

    while True:
        for event in pg.event.get():
            if event.type == QUIT or (event.type == KEYDOWN
                                      and event.key == K_ESCAPE):
                pg.quit()
                sys.exit()
            if event.type == KEYDOWN and (event.key == K_SPACE
                                          or event.key == K_UP):
                return {
                    'playery': playery + playerShmVals['val'],
                    'basex': basex,
                    'playerIndexGen': playerIndexGen
                }

        # 控制草地移动
        if (loopIter + 1) % 5 == 0:
            playerIndex = next(playerIndexGen)
        loopIter = (loopIter + 1) % 30
        basex = -((-basex + 4) % baseShift)
        # 控制鸟上下飞行
        playerShm(playerShmVals)

        # draw
        SCREEN.blit(IMAGES['background'], IMAGES['background'].get_rect())
        # fill background with
        #SCREEN.fill((141, 238, 238))
        SCREEN.blit(IMAGES['player'][playerIndex],
                    (playerx, playery + playerShmVals['val']))
        SCREEN.blit(IMAGES['img_ground'], (basex, BASTY))

        pg.display.update()
        #  FPSCLOCK.tick(FPS)


def game(movementInfo):
    score = playerIndex = loopIter = 0
    playerIndexGen = movementInfo['playerIndexGen']
    playerx, playery = int(SCREENWIDTH * 0.1), movementInfo['playery']

    basex = movementInfo['basex']
    baseShift = IMAGES['img_ground'].get_width() - SCREENWIDTH

    # get 2 new pipes to add upperTrees and lowerTrees list
    newTree1 = getRandomTree()
    newTree2 = getRandomTree()
    newTree3 = getRandomTree()
    newTree4 = getRandomTree()
    upperTrees = [{
        'x': SCREENWIDTH - DIS * 3,
        'y': newTree1[0]['y']
    }, {
        'x': SCREENWIDTH - DIS * 2,
        'y': newTree2[0]['y']
    }, {
        'x': SCREENWIDTH - DIS,
        'y': newTree3[0]['y']
    }, {
        'x': SCREENWIDTH,
        'y': newTree4[0]['y']
    }]
    lowerTrees = [{
        'x': SCREENWIDTH - DIS * 3,
        'y': newTree1[1]['y']
    }, {
        'x': SCREENWIDTH - DIS * 2,
        'y': newTree2[1]['y']
    }, {
        'x': SCREENWIDTH - DIS,
        'y': newTree3[1]['y']
    }, {
        'x': SCREENWIDTH,
        'y': newTree4[1]['y']
    }]

    treeVelX = -12.5

    # player's velocity along Y,default same as playerFlapped
    playerVelY = -9
    # max vel along Y,max descend speed
    playerMaxVelY = 30
    # min vel along Y,min descend speed
    playerMinVelY = -8
    # player downward accleration
    playerAccY = 4
    # player's rotation
    playerRot = 45
    # angular speed
    playerVelRot = 3
    # rotation threshold
    playerRotThr = 20
    # player's speed on flapping
    playerFlapAcc = -25
    playerFlapped = False

    count = 0

    while True:
        for event in pg.event.get():
            if event.type == QUIT or (event.type == KEYDOWN
                                      and event.key == K_ESCAPE):
                pg.quit()
                sys.exit()

            if event.type == KEYDOWN and (event.key == K_SPACE
                                          or event.key == K_UP):
                if playery > -2 * IMAGES['player'][0].get_height():
                    playerVelY = playerFlapAcc
                    playerFlapped = True

        # check for crash
        crashTest = checkCrash({
            'x': playerx,
            'y': playery,
            'index': playerIndex
        }, upperTrees, lowerTrees)
        if crashTest[0]:
            return {
                'y': playery,
                'groundCrash': crashTest[1],
                'basex': basex,
                'upperTrees': upperTrees,
                'lowerTrees': lowerTrees,
                'score': score,
                'playerVelY': playerVelY,
                'playerRot': playerRot
            }
        #  总距离
        playerMidPos = playerx + IMAGES['player'][0].get_width() / 2
        for tree in upperTrees:
            treeMidPos = tree['x'] + IMAGES['tree'][0].get_width() / 2
            if treeMidPos <= playerMidPos < treeMidPos + 4:
                score += 1

        # 移动草地
        if (loopIter + 1) % 3 == 0:
            playerIndex = next(playerIndexGen)

        loopIter = (loopIter + 1) % 30
        basex = -((-basex + 200) % baseShift)

        if playerRot > -90:
            playerRot -= playerVelRot

        # 鸟的移动
        if playerVelY < playerMaxVelY and not playerFlapped:
            playerVelY += playerAccY
        if playerFlapped:
            playerFlapped = False

            playerRot = 45

        playerHeight = IMAGES['player'][playerIndex].get_height()
        playery += min(playerVelY, BASTY - playery - playerHeight)

        # 移动树
        for uTree, lTree in zip(upperTrees, lowerTrees):
            uTree['x'] += treeVelX
            lTree['x'] += treeVelX

        # 产生新的树
        count = (count + treeVelX) % DIS
        if count == 0:
            newTree = getRandomTree()
            upperTrees.append(newTree[0])
            lowerTrees.append(newTree[1])

        # 移除最前面一棵树
        if upperTrees[0]['x'] < -IMAGES['tree'][0].get_width():
            upperTrees.pop(0)
            lowerTrees.pop(0)

        # 移动背景
        SCREEN.blit(IMAGES['background'], (0, 0))

        for uTree, lTree in zip(upperTrees, lowerTrees):
            SCREEN.blit(IMAGES['tree'][0], (uTree['x'], uTree['y']))
            SCREEN.blit(IMAGES['tree'][1], (lTree['x'], lTree['y']))

        SCREEN.blit(IMAGES['img_ground'], (basex, BASTY))
        '''visibleRot = min(playerRotThr, playerRot)

        playerSurface = pg.transform.rotate(IMAGES['player'][playerIndex],
                                            visibleRot)'''

        SCREEN.blit(IMAGES['player'][playerIndex], (playerx, playery))

        pg.display.update()
        # FPSCLOCK.tick(FPS)


def gameOver(crashInfo):
    '''鸟坠落游戏结束'''
    score = crashInfo['score']
    playerx = SCREENWIDTH * 0.1
    playery = crashInfo['y']
    playerHeight = IMAGES['player'][0].get_height()
    playerVelY = crashInfo['playerVelY']
    playerAccY = 2
    playerRot = crashInfo['playerRot']
    playerVelRot = 7

    basex = crashInfo['basex']

    upperTrees, lowerTrees = crashInfo['upperTrees'], crashInfo['lowerTrees']

    while True:
        for event in pg.event.get():
            if event.type == QUIT or (event.type == KEYDOWN
                                      and event.key == K_ESCAPE):
                pg.quit()
                sys.exit()
            if event.type == KEYDOWN and (event.key == K_SPACE
                                          or event.key == K_UP):
                if playery + playerHeight >= BASTY - 1:
                    return

        # 调整y值
        if playery + playerHeight < BASTY - 1:
            playery += min(playerVelY, BASTY - playery - playerHeight)
        if playerVelY < 15:
            playerVelY += playerAccY

        if not crashInfo['groundCrash']:
            if playerRot > -90:
                playerRot -= playerVelRot

        SCREEN.blit(IMAGES['background'], (0, 0))

        for uTree, lTree in zip(upperTrees, lowerTrees):
            SCREEN.blit(IMAGES['tree'][0], (uTree['x'], uTree['y']))
            SCREEN.blit(IMAGES['tree'][1], (lTree['x'], lTree['y']))

        SCREEN.blit(IMAGES['img_ground'], (basex, BASTY))

        playerSurface = pg.transform.rotate(IMAGES['player'][1], playerRot)
        SCREEN.blit(IMAGES['player'][1], (playerx, playery))

        # FPSCLOCK.tick(FPS)
        pg.display.update()


def checkCrash(player, upperTrees, lowerTrees):
    '''如果坠地或撞到树就返回True'''
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][1].get_height()

    # 撞到地面
    if player['y'] + player['h'] >= BASTY - 1:
        return [True, True]
    else:
        playerRect = pg.Rect(player['x'], player['y'], player['w'],
                             player['h'])
        treeW = IMAGES['tree'][0].get_width()
        treeH = IMAGES['tree'][0].get_height()

        for uTree, lTree in zip(upperTrees, lowerTrees):
            uTreeRect = pg.Rect(uTree['x'], uTree['y'], treeW, treeH)
            lTreeRect = pg.Rect(lTree['x'], lTree['y'], treeW, treeH)

            pHitMask = HITMASKS['player'][pi]
            uHitMask = HITMASKS['tree'][0]
            lHitMask = HITMASKS['tree'][1]

            # 撞到树
            uCollide = pixelCollision(playerRect, uTreeRect, pHitMask,
                                      uHitMask)
            lCollide = pixelCollision(playerRect, lTreeRect, pHitMask,
                                      lHitMask)

            if uCollide or lCollide:
                return [True, False]
    return [False, False]


def getRandomTree():
    '''随机返回一棵树'''
    gapY = random.randrange(0, int(BASTY * 0.6 - TREEGAPSIZE))
    gapY += int(BASTY * 0.2)
    treeHeight = IMAGES['tree'][0].get_height()
    treeX = SCREENWIDTH
    return [
        # upper tree
        {
            'x': treeX,
            'y': gapY - treeHeight
        },
        # lower tree
        {
            'x': treeX,
            'y': gapY + TREEGAPSIZE
        }
    ]


def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    '''Check if two objects collide and not just their rects'''
    rect = rect1.clip(rect2)
    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y
    for i in range(rect.width):
        for j in range((rect.height)):
            if hitmask1[x1 + i][y1 + j] and hitmask2[x2 + i][y2 + j]:
                return True
    return False


def playerShm(playerShm):
    '''oscillates(震荡) the value of playerShm['val'] between 8 and -8'''
    if abs(playerShm['val']) == 8:
        playerShm['dir'] *= -1

    if playerShm['dir'] == 1:
        playerShm['val'] += 1
    else:
        playerShm['val'] -= 1


def getHitmask(image):
    '''return a hitmask using an image's alpha'''
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x, y))[3]))
    return mask


if __name__ == '__main__':
    main()
