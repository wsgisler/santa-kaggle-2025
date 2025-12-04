# Santa Kaggle 2025 - A special case of polygon packing

This years Santa kaggle problem is about packing a number n of predefined christmas tree shaped polygons into the smallest possible square. Trees can touch, but not overlap. The initial polygon is defined as follows, and every polygon is identical:

```
trunk_w = Decimal('0.15')
trunk_h = Decimal('0.2')
base_w = Decimal('0.7')
mid_w = Decimal('0.4')
top_w = Decimal('0.25')
tip_y = Decimal('0.8')
tier_1_y = Decimal('0.5')
tier_2_y = Decimal('0.25')
base_y = Decimal('0.0')
trunk_bottom_y = -trunk_h

initial_polygon = Polygon(
	[
		# Start at Tip
		(Decimal('0.0') * scale_factor, tip_y * scale_factor),
		# Right side - Top Tier
		(top_w / Decimal('2') * scale_factor, tier_1_y * scale_factor),
		(top_w / Decimal('4') * scale_factor, tier_1_y * scale_factor),
		# Right side - Middle Tier
		(mid_w / Decimal('2') * scale_factor, tier_2_y * scale_factor),
		(mid_w / Decimal('4') * scale_factor, tier_2_y * scale_factor),
		# Right side - Bottom Tier
		(base_w / Decimal('2') * scale_factor, base_y * scale_factor),
		# Right Trunk
		(trunk_w / Decimal('2') * scale_factor, base_y * scale_factor),
		(trunk_w / Decimal('2') * scale_factor, trunk_bottom_y * scale_factor),
		# Left Trunk
		(-(trunk_w / Decimal('2')) * scale_factor, trunk_bottom_y * scale_factor),
		(-(trunk_w / Decimal('2')) * scale_factor, base_y * scale_factor),
		# Left side - Bottom Tier
		(-(base_w / Decimal('2')) * scale_factor, base_y * scale_factor),
		# Left side - Middle Tier
		(-(mid_w / Decimal('4')) * scale_factor, tier_2_y * scale_factor),
		(-(mid_w / Decimal('2')) * scale_factor, tier_2_y * scale_factor),
		# Left side - Top Tier
		(-(top_w / Decimal('4')) * scale_factor, tier_1_y * scale_factor),
		(-(top_w / Decimal('2')) * scale_factor, tier_1_y * scale_factor),
	]
)
```

Trees can be rotated and can be located at any x or y position. For every n between 1 and 200, the best positions of trees need to be found. The best position of trees minimizes the bounding square of these polygons.

## Comments

I unfortunately didn't have enough time to really go deep, but I tried a couple of approaches. My solutions are good (getting a score of about 73 - current best score 69.5), but still too far away from the best submissions to be competitive. Better solutions would require a more refined approach and a bigger time investment.

All of my code is vibe coded to a large extent: I described my approach as detailed as possible and let Claude Code do the work. While I have worked like this before for simpler things, this is the first time that I use this kind of approach for complex algorithms. I was genuinely surprised how well this worked, despite the complexity of the work.

My approach combines multiple approaches. All files starting with "solver_" are solvers that can be used to generate solutions. There are simulated annealing approaches, genetic approaches, some approaches that utilize OR Tools to solve this in a similar way how a cutting stock problem would be solved, a special simulated annealing algorithm that generates repeating patterns and even an exact approach using Gurobi (which turned out to be very bad with terrible bounds). For efficiency reasons, the simulated annealing approach was translated to Rust (automatically, using Claude).

The best solutions are then combined (tools_integrate_solutions.py) and then solver_combine_solutions.py is run to find improved solutions from other solutions (e.g. leaving away one random polygon from the 200 tree solution might find a 199 tree solution that is better than the current best solution for 199 trees). Finally tools_combine_solutions.py is executed to create the submission file.

It was fun to develop this, but somehow also frustrating to compete in a competition where people openly shared their notebooks and even post final solution files. So, it was easy for new competitors to just submit other people's work or do minor tweaks on existing approaches. Everything that you find in this repository is my own work, and not in any way copied from anyone else. Due to the large number of "copycat" submissions, I decided to pull the plug at some point and not commit more time to this.

Hopefully this is useful for some people. Have fun!