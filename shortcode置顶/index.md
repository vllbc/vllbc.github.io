# shortcode(置顶)

贴一下可以玩的shortcode。

## 音乐播放
### 播放列表
夏日口袋专辑：

{{< music auto="https://music.163.com/album?id=73470837&uct2=U2FsdGVkX18gTMY/Tb1+2PmOZr2G/Q7mOdM/mANJ8xY=" >}}


### 播放单曲

最爱的一首（我是紬厨）：
{{< music netease song 1311346841 >}}



## 视频播放
### bilibili
{{< bilibili BV1ptXPYREe7 1>}} 
有多P可以选择集数

### youtube
{{< youtube embed="GEXa2yrAucM" >}}

## admonition
类型有：note、abstract、info、tip、success、question、warning、failure、danger、bug、example、quote。
{{< admonition type=tip  open=true >}}
一个 **技巧** 横幅
{{< /admonition >}}



## typeit

### 简单内容
{{< typeit >}}
这一个带有基于 [TypeIt](https://typeitjs.com/) 的 **打字动画** 的 *段落*...
{{< /typeit >}}

### 代码内容

{{< typeit code=python >}}
def main():
    print('hello world')
main()
{{< /typeit >}}


##  mapbox
### 默认样式

{{< mapbox 121.485 31.233 12 >}}


### 自定义样式

{{< mapbox lng=-122.252 lat=37.453 zoom=10 marked=false light-style="mapbox://styles/mapbox/streets-zh-v1" >}}

