-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.

local SingleGoal, parent = torch.class('SingleGoal', 'MazeBase')

function SingleGoal:__init(opts, vocab)
    parent.__init(self, opts, vocab)
    self:add_default_items()
    -- self.goal = self:place_item_rand({type = 'goal', name = 'goal' .. 1})
    self.goal = self:place_item({type = 'goal', name = 'goal' .. 1}, 3,4)
    self:place_item({type = 'cow'}, 6, 8)
    self:place_item({type = 'heart'}, 8, 6)
    self:place_item({type = 'cow'}, 2, 4)
    self:place_item({type = 'duck'}, 4, 2)
    self:place_item({type = 'heart'}, 1, 8)
    self:place_item({type = 'tree'}, 3, 6)
    self:place_item({type = 'heart'}, 5, 4)
    self:place_item({type = 'duck'}, 7, 2)
    self:place_item({type = 'heart'}, 6, 1)
    self:place_item({type = 'heart'}, 8, 3)
    self:place_item({type = 'duck'}, 2, 5)
    self:place_item({type = 'tree'}, 4, 7)
end

function SingleGoal:update()
    parent.update(self)
    if not self.finished then
        if self.goal.loc.y == self.agent.loc.y and self.goal.loc.x == self.agent.loc.x then
            self.finished = true
        end
    end
end

function SingleGoal:get_reward()
    if self.finished then
        return -self.costs.goal
    else
        return parent.get_reward(self)
    end
end
