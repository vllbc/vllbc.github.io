# shortcode(置顶)

贴一下可以玩的shortcode。

## 音乐播放
### 播放列表

{{< music "https://music.163.com/#/playlist?id=60198" >}}


### 播放单曲

{{< music server="netease" type="song" id="1868553" >}}


## 视频播放
### bilibili
{{< bilibili BV1144y167iZ 1>}} 
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
{{< mapbox lng=121.485 lat=31.233 zoom=12 >}}

### 自定义样式

{{< mapbox lng=-122.252 lat=37.453 zoom=10 marked=false light-style="mapbox://styles/mapbox/streets-zh-v1" >}}

