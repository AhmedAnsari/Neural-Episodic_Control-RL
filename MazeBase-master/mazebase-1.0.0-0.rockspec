package = "mazebase"
version = "1.0.0-0"

source = {
  dir = "MazeBase-master",
  url = "https://github.com/facebook/MazeBase/archive/master.zip",
}

description = {
  summary = "Game based library for reinforcement learning",
  homepage = "https://github.com/facebook/MazeBase",
  license = "BSD",
  maintainer = "",
}

build = {
  type = "builtin",
  modules = {
    mazebase = "lua/mazebase/init.lua",
  },
}
