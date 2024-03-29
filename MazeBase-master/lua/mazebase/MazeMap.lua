-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.

require('image')
local fpath = require 'sys.fpath'

local MazeMap = torch.class('MazeMap')

function MazeMap:__init(opts)
    self.height = opts.map_height or 10
    self.width = opts.map_width or 10
    self.img_path = paths.concat(fpath(), 'images')

    -- Items by x,y location
    self.items = {}
    for y = 1, self.height do
        self.items[y] = {}
        for x = 1, self.width do
            self.items[y][x] = {}
        end
    end

    self.visibility_mask = torch.Tensor(self.height, self.width)
    self.visibility_mask:fill(1)
end

function MazeMap:add_item(item)
    table.insert(self.items[item.loc.y][item.loc.x], item)
end

function MazeMap:remove_item(item)
    local l = self.items[item.loc.y][item.loc.x]
    for i = 1, #l do
        if l[i].id == item.id then
            table.remove(l, i)
            break
        end
    end
end

function MazeMap:get_empty_loc(fat)
    local fat = fat or 0
    local x, y
    for i = 1, 100 do
        y = torch.random(1+fat, self.height-fat)
        x = torch.random(1+fat, self.width-fat)
        local empty = true
        for j, e in pairs(self.items[y][x]) do
            if not e.attr._immaterial then
                empty = false
            end
        end
        if empty then return y, x end
    end
    error('failed 100 times to find empty location')
end

function MazeMap:is_loc_reachable(y, x)
     if y < 1 or x < 1 then
        return false
    elseif y > self.height or x > self.width then
        return false
    end
    local l = self.items[y][x]
    local is_reachable = true
    for i = 1, #l do
        is_reachable = is_reachable and l[i]:is_reachable()
    end
    return is_reachable
end

function MazeMap:is_loc_visible(y, x)
    if self.visibility_mask[y][x] == 1 then
        return true
    else
        return false
    end
end

function MazeMap:to_image()
    local K = 32
    local img = torch.Tensor(3, self.height * K, self.width * K):fill(1)
    local img_goals={}
    for s = 1, 9 do
        img_goals[s]= image.load(self.img_path .. '/goal' .. s .. '.png')
    end
    local img_block = image.load(self.img_path .. '/block.png')
    local img_water = image.load(self.img_path .. '/water.png')
    local img_pushableblock = image.load(self.img_path .. '/pushableblock.png')
    local img_bfire = image.load(self.img_path .. '/blue_fire.png')
    local img_rfire = image.load(self.img_path .. '/red_fire.png')
    -- local img_agent = image.load(self.img_path .. '/agent.png')
    local img_agent = image.load(self.img_path .. '/agent.jpg')
    local img_box = image.load(self.img_path .. '/box.jpg')
    local img_candy = image.load(self.img_path .. '/candy.jpg')
    local img_cow = image.load(self.img_path .. '/cow.jpg')
    local img_diamond = image.load(self.img_path .. '/diamond.jpg')
    local img_duck = image.load(self.img_path .. '/duck.jpg')
    local img_egg = image.load(self.img_path .. '/egg.jpg')
    local img_enemy = image.load(self.img_path .. '/enemy.jpg')
    local img_heart = image.load(self.img_path .. '/heart.jpg')
    local img_meat = image.load(self.img_path .. '/meat.jpg')
    local img_milk = image.load(self.img_path .. '/milk.jpg')
    local img_pig = image.load(self.img_path .. '/pig.jpg')
    local img_rock = image.load(self.img_path .. '/rock.jpg')
    local img_stone = image.load(self.img_path .. '/stone.jpg')
    local img_tree = image.load(self.img_path .. '/tree.jpg')
    local img_wood = image.load(self.img_path .. '/wood.jpg')

    local img_starenemy = {}
    img_starenemy[1] = image.load(self.img_path .. '/starenemy1.png')
    img_starenemy[2] = image.load(self.img_path .. '/starenemy2.png')
    img_starenemy[3] = image.load(self.img_path .. '/starenemy3.png')
    img_starenemy[4] = image.load(self.img_path .. '/starenemy4.png')
    img_starenemy[5] = image.load(self.img_path .. '/starenemy5.png')

    for y = 1, self.height do
        for x = 1, self.width do
            local c = img:narrow(2,1+(y-1)*K,K):narrow(3,1+(x-1)*K,K)
            for i = 1, #self.items[y][x] do
                local item = self.items[y][x][i]
                if not item.attr._invisible then
                    if item.type == 'block' then
                        c:copy(img_block)
                    elseif item.type == 'pushableblock' then
                        c:copy(img_pushableblock)
            		elseif item.type == 'water' then
                        c:copy(img_water)
                    --dmittal
                    elseif item.type == 'box' then
                        c:copy(img_box)
                    elseif item.type == 'candy' then
                        c:copy(img_candy)
                    elseif item.type == 'cow' then
                        c:copy(img_cow)
                    elseif item.type == 'diamond' then
                        c:copy(img_diamond)
                    elseif item.type == 'duck' then
                        c:copy(img_duck)
                    elseif item.type == 'egg' then
                        c:copy(img_egg)
                    elseif item.type == 'enemy' then
                        c:copy(img_enemy)
                    elseif item.type == 'heart' then
                        c:copy(img_heart)
                    elseif item.type == 'meat' then
                        c:copy(img_meat)
                    elseif item.type == 'milk' then
                        c:copy(img_milk)
                    elseif item.type == 'pig' then
                        c:copy(img_pig)
                    elseif item.type == 'rock' then
                        c:copy(img_rock)
                    elseif item.type == 'stone' then
                        c:copy(img_stone)
                    elseif item.type == 'tree' then
                        c:copy(img_tree)
                    elseif item.type == 'wood' then
                        c:copy(img_wood)
                    
                    elseif item.type == 'goal' then
                        for a = 1, 9 do
                            if item.name == 'goal' .. a then
                                c:copy(img_goals[a])
                            end
                        end
                    elseif item.type == 'door' then
                         if item.attr._c == 1 then
                            c[1]:fill(0)
                        elseif item.attr._c == 2 then
                            c[2]:fill(0)
                        elseif item.attr._c == 3 then
                            c[3]:fill(0)
                        else
                            c:fill(0.5)
                        end
                         if item.attr.open=='open' then
                             c[1]:narrow(2,1,K/2):fill(0)
                             c[2]:narrow(2,1,K/2):fill(1)
                             c[3]:narrow(2,1,K/2):fill(0)
                         else
                             c[1]:narrow(2,1,K/2):fill(1)
                             c[2]:narrow(2,1,K/2):fill(0)
                             c[3]:narrow(2,1,K/2):fill(0)
                         end
                    elseif item.type == 'agent' then
                        c:copy(img_agent)
                    elseif item.type == 'StarEnemy' then
                        c[1]:fill(1)
                        c[2]:fill(0)
                        c[3]:fill(0)
                        for a = 1, 5 do
                            if item.name == 'enemy' .. a then
                                c:copy(img_starenemy[a])
                            end
                        end
                    elseif item.type == 'switch' then
                        if item.attr._c == 1 then
                            c[1]:fill(0)
                        elseif item.attr._c == 2 then
                            c[2]:fill(0)
                        elseif item.attr._c == 3 then
                            c[3]:fill(0)
                        else
                            c:fill(0.5)
                        end
                    elseif item.type == 'crumb' then
                        c[{ 1, {15,18}, {15,18} }]:fill(1)
                        c[{ 2, {15,18}, {15,18} }]:fill(0)
                        c[{ 3, {15,18}, {15,18} }]:fill(0)
                    end
                end
            end
            if self.ygoal and self.ygoal == y and self.xgoal == x then
                c[2]:add(.4)
            end
        end
    end
    if self.agent_shots_d then
        for s = 1, #self.agent_shots_d do
            local l = self.agent_shots_d[s]
            for t = 1, l:size(1) do
                local y = l[t][1]
                local x = l[t][2]
                local c = img:narrow(2,1+(y-1)*K,K):narrow(3,1+(x-1)*K,K)
                c:copy(img_rfire:sub(1,3,1,-1,1,-1))
            end
        end
    end
    if self.enemy_shots_d then
        for s = 1, #self.enemy_shots_d do
            local l = self.enemy_shots_d[s]
            for t = 1, l:size(1) do
                local y = l[t][1]
                local x = l[t][2]
                local c = img:narrow(2,1+(y-1)*K,K):narrow(3,1+(x-1)*K,K)
                c:copy(img_bfire:sub(1,3,1,-1,1,-1))
            end
        end
    end

    local mask = self.visibility_mask:view(1, self.height, 1, self.width, 1)
    mask = mask:expand(3, self.height, K, self.width, K):clone()
    mask = mask:view(3, self.height * K, self.width * K)
    img:cmul(mask)
    return img
end
