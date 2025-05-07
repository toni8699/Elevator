import random
import time
import argparse
from collections import defaultdict

NUM_FLOORS = 29
GROUND_FLOOR = 0
TIME_PER_FLOOR_TRAVEL = 1
DOOR_CYCLE_AND_BOARD_TIME = 1

# --- Global Configuration ---
G_RETURN_ENABLED = False
OPPORTUNISTIC_PICKUP_ENABLED = False

# --- Time Definitions ---
MORNING_PEAK_START = 7 * 60;
MORNING_PEAK_END = 9 * 60
EVENING_PEAK_START = 17 * 60;
EVENING_PEAK_END = 19 * 60
LOW_TRAFFIC_START = 0 * 60;
LOW_TRAFFIC_END = 7 * 60

PROB_LOW_TRAFFIC = 0.05;
PROB_NORMAL_TRAFFIC = 0.2;
PROB_PEAK_TRAFFIC = 0.5


class Elevator:
    def __init__(self, id_num, is_designated_g_return_elevator=False):
        self.id = f"E{id_num}"
        self.current_floor = GROUND_FLOOR
        self.state = "IDLE"
        self.is_designated_g_return_elevator = is_designated_g_return_elevator
        self.time_in_current_state = 0
        self.passengers_inside = {}
        self.committed_pickups = {}
        self.current_direction_of_service = None
        self.next_target_stop = None
        self.non_idle_time_steps = 0
        self.empty_g_return_trips_initiated = 0

    def __repr__(self):
        g_ret_marker = f"(G-Ret:{'ON' if self.is_designated_g_return_elevator and G_RETURN_ENABLED else 'OFF'})" 
        opp_marker = f"(Opp:{'ON' if OPPORTUNISTIC_PICKUP_ENABLED else 'OFF'})" 
        pass_dests = sorted([p['dest'] for p in self.passengers_inside.values()])
        pickup_origins = sorted([p['origin'] for p in self.committed_pickups.values()])
        return (f"{self.id} " 
                f"F:{self.current_floor} S:{self.state} Dir:{self.current_direction_of_service} Target:{self.next_target_stop} "
                f"PassDests:{pass_dests} PickupOrigins:{pickup_origins}")

    def is_effectively_idle(self):
        return self.state == "IDLE" and not self.passengers_inside and not self.committed_pickups

    def has_tasks(self):
        return bool(self.passengers_inside or self.committed_pickups)

    def add_assigned_hall_call(self, req_id, origin, dest, direction):
        self.committed_pickups[req_id] = {'origin': origin, 'dest': dest, 'direction': direction}
        if self.is_effectively_idle():  
            self._determine_next_target_and_direction()

    def _determine_next_target_and_direction(self):
        if not self.has_tasks():
            self.current_direction_of_service = None
            self.next_target_stop = None
            if self.state != "DOOR_CYCLE": self.state = "IDLE"
            return

        potential_stops = defaultdict(lambda: {'dropoff': [], 'pickup_up': [], 'pickup_down': []})
        for req_id, p_data in self.passengers_inside.items(): potential_stops[p_data['dest']]['dropoff'].append(req_id)
        for req_id, c_data in self.committed_pickups.items():
            if c_data['direction'] == "UP":
                potential_stops[c_data['origin']]['pickup_up'].append(req_id)
            elif c_data['direction'] == "DOWN":
                potential_stops[c_data['origin']]['pickup_down'].append(req_id)

        determined_target = None
        determined_direction = self.current_direction_of_service  

        if determined_direction == "UP":
            for floor in sorted(potential_stops.keys()):
                if floor >= self.current_floor and (
                        potential_stops[floor]['dropoff'] or potential_stops[floor]['pickup_up']):
                    determined_target = floor;
                    break
        elif determined_direction == "DOWN":
            for floor in sorted(potential_stops.keys(), reverse=True):
                if floor <= self.current_floor and (
                        potential_stops[floor]['dropoff'] or potential_stops[floor]['pickup_down']):
                    determined_target = floor;
                    break

        if determined_target is None:
            determined_direction = "UP"
            for floor in sorted(potential_stops.keys()):
                if floor >= self.current_floor and (
                        potential_stops[floor]['dropoff'] or potential_stops[floor]['pickup_up']):
                    determined_target = floor;
                    break

            if determined_target is None:  
                determined_direction = "DOWN"
                for floor in sorted(potential_stops.keys(), reverse=True):
                    if floor <= self.current_floor and (
                            potential_stops[floor]['dropoff'] or potential_stops[floor]['pickup_down']):
                        determined_target = floor;
                        break

            if determined_target is None:  
                determined_direction = None  
                min_dist = float('inf')
                for floor, tasks in potential_stops.items():
                    if tasks['dropoff'] or tasks['pickup_up'] or tasks['pickup_down']:
                        dist = abs(self.current_floor - floor)
                        if dist < min_dist:
                            min_dist = dist
                            determined_target = floor
                            if tasks['pickup_up'] or (tasks['dropoff'] and floor > self.current_floor):
                                determined_direction = "UP"
                            elif tasks['pickup_down'] or (tasks['dropoff'] and floor < self.current_floor):
                                determined_direction = "DOWN"
                            elif tasks['dropoff'] and floor == self.current_floor:  
                                determined_direction = self.current_direction_of_service  

        self.next_target_stop = determined_target
        self.current_direction_of_service = determined_direction if determined_target is not None else None

        if self.next_target_stop is not None:
            if self.current_floor == self.next_target_stop:
                self.state = "DOOR_CYCLE"; self.time_in_current_state = 0
            elif self.next_target_stop > self.current_floor:
                self.current_floor += 1; self.current_direction_of_service = "UP"
            elif self.next_target_stop < self.current_floor:
                self.current_floor -= 1; self.current_direction_of_service = "DOWN"


    def step(self, building_stats_collector, current_sim_time):
        if self.state != "IDLE" or self.has_tasks(): self.non_idle_time_steps += 1
        self.time_in_current_state += 1

        if self.state == "IDLE":
            if not self.has_tasks():
                if G_RETURN_ENABLED and self.is_designated_g_return_elevator and self.current_floor != GROUND_FLOOR:
                    self.current_direction_of_service = "DOWN" if self.current_floor > GROUND_FLOOR else "UP"
                    if self.current_floor == GROUND_FLOOR:
                        self.current_direction_of_service = None; self.state = "IDLE"
                    else:
                        self.next_target_stop = GROUND_FLOOR; self.state = "MOVING"
                    self.empty_g_return_trips_initiated += 1
            else:
                self._determine_next_target_and_direction()
            return

        elif self.state == "DOOR_CYCLE":
            if self.time_in_current_state >= DOOR_CYCLE_AND_BOARD_TIME:
                dropped_ids = [];
                picked_ids = []
                for r, p in list(self.passengers_inside.items()):
                    if p['dest'] == self.current_floor: building_stats_collector.record_trip_completion(r,
                                                                                                        current_sim_time,
                                                                                                        p[
                                                                                                            'pickup_time']); dropped_ids.append(
                        r)
                for r in dropped_ids: del self.passengers_inside[r]
                for r, c in list(self.committed_pickups.items()):
                    if c['origin'] == self.current_floor:
                        if self.current_direction_of_service is None or c[
                            'direction'] == self.current_direction_of_service or not self.passengers_inside:
                            if self.current_direction_of_service is None and not self.passengers_inside: self.current_direction_of_service = \
                            c['direction']
                            self.passengers_inside[r] = {'origin': c['origin'], 'dest': c['dest'],
                                                         'pickup_time': current_sim_time};
                            picked_ids.append(r)
                for r in picked_ids: del self.committed_pickups[r]
                self.time_in_current_state = 0
                self._determine_next_target_and_direction()
                if not self.has_tasks() and self.state != "MOVING": self.state = "IDLE"  

        elif self.state == "MOVING":
            if self.next_target_stop is None: self._determine_next_target_and_direction()
            if self.next_target_stop is None: self.state = "IDLE"; return  

            if self.current_floor == self.next_target_stop:
                self.state = "DOOR_CYCLE"; self.time_in_current_state = 0
            elif self.next_target_stop > self.current_floor:
                self.current_floor += 1; self.current_direction_of_service = "UP"
            elif self.next_target_stop < self.current_floor:
                self.current_floor -= 1; self.current_direction_of_service = "DOWN"


class StatsCollector:  
    def __init__(self):
        self.total_wait_time = 0;
        self.max_wait_time = 0
        self.total_trip_time = 0;
        self.requests_completed = 0
        self.request_creation_times = {};
        self.request_assignment_times = {}
        self.wait_times_by_period = defaultdict(list);
        self.trip_times_by_period = defaultdict(list)

    def get_traffic_period(self, t):
        if MORNING_PEAK_START <= t < MORNING_PEAK_END or EVENING_PEAK_START <= t < EVENING_PEAK_END:
            return "peak"
        elif LOW_TRAFFIC_START <= t < LOW_TRAFFIC_END:
            return "low"
        else:
            return "normal"

    def log_request_creation(self, r, t):
        self.request_creation_times[r] = t

    def log_request_assignment(self, r, t):
        self.request_assignment_times[r] = t;
        wt = t - self.request_creation_times.get(r, t)
        self.total_wait_time += wt;
        self.max_wait_time = max(self.max_wait_time, wt)
        self.wait_times_by_period[self.get_traffic_period(self.request_creation_times.get(r, t))].append(wt)

    def record_trip_completion(self, r, dt, pt):
        if pt is None: return
        tt = dt - pt;
        self.total_trip_time += tt;
        self.requests_completed += 1
        self.trip_times_by_period[self.get_traffic_period(self.request_creation_times.get(r, dt))].append(tt)

    def get_average_wait_time(self, p=None):
        if p: ts = self.wait_times_by_period[p]; return sum(ts) / len(ts) if ts else 0
        num_assigned = len(self.request_assignment_times)
        return self.total_wait_time / num_assigned if num_assigned > 0 else 0

    def get_average_trip_time(self, p=None):
        if p: ts = self.trip_times_by_period[p]; return sum(ts) / len(ts) if ts else 0
        return self.total_trip_time / self.requests_completed if self.requests_completed > 0 else 0

    def print_period_stats(self):
        print("\nPerformance by Traffic Period:")
        for p in ["low", "normal", "peak"]:
            print(
                f"  {p.capitalize()} Traffic ({len(self.wait_times_by_period[p])} reqs): Avg Wait: {self.get_average_wait_time(p):.2f} min, Avg Trip: {self.get_average_trip_time(p):.2f} min")


class Building:
    def __init__(self, num_elevators=3):
        self.elevators = [Elevator(i + 1, (i == 0)) for i in range(num_elevators)]
        self.pending_hall_calls = []
        self.stats = StatsCollector()
        self.total_requests_generated_this_simulation = 0

    def add_request(self, start_floor, dest_floor, direction, current_time_step, req_id):
        self.pending_hall_calls.append(
            {'origin': start_floor, 'dest': dest_floor, 'direction': direction, 'req_id': req_id,
             'time': current_time_step})
        self.stats.log_request_creation(req_id, current_time_step)

    def assign_requests(self, current_time_step):
        calls_processed_this_step = []

        if OPPORTUNISTIC_PICKUP_ENABLED:
            for call_data in list(self.pending_hall_calls):
                if call_data in calls_processed_this_step: continue  

                call_origin = call_data['origin']
                call_direction = call_data['direction']
                call_dest = call_data['dest']
                req_id = call_data['req_id']
                best_opportunistic_elevator = None
                min_dist_to_pickup = float('inf')

                for elev in self.elevators:
                    if not elev.current_direction_of_service or elev.current_direction_of_service == call_direction:
                        can_pickup = False
                        if call_direction == "UP" and elev.current_floor <= call_origin:
                            can_pickup = True  
                        elif call_direction == "DOWN" and elev.current_floor >= call_origin:
                            can_pickup = True  

                        if elev.is_effectively_idle() and not elev.has_tasks():
                            can_pickup = True

                        if can_pickup:
                            dist = abs(elev.current_floor - call_origin)
                            if dist < min_dist_to_pickup:
                                min_dist_to_pickup = dist
                                best_opportunistic_elevator = elev

                if best_opportunistic_elevator:
                    best_opportunistic_elevator.add_assigned_hall_call(req_id, call_origin, call_dest, call_direction)
                    self.stats.log_request_assignment(req_id, current_time_step)
                    calls_processed_this_step.append(call_data)

        for processed_call in calls_processed_this_step:
            if processed_call in self.pending_hall_calls: self.pending_hall_calls.remove(processed_call)

        for call_data in list(self.pending_hall_calls):  
            if call_data in calls_processed_this_step: continue  

            call_origin = call_data['origin']
            call_direction = call_data['direction']
            call_dest = call_data['dest']
            req_id = call_data['req_id']
            best_standard_elevator = None
            min_metric_standard = float('inf')

            idle_candidates = [e for e in self.elevators if e.is_effectively_idle()]
            if idle_candidates:
                for elev in idle_candidates:
                    dist = abs(elev.current_floor - call_origin)
                    metric = dist
                    if G_RETURN_ENABLED and elev.is_designated_g_return_elevator and len(
                        idle_candidates) > 1: metric += 0.1
                    if metric < min_metric_standard:
                        min_metric_standard = metric
                        best_standard_elevator = elev

            if best_standard_elevator:
                best_standard_elevator.add_assigned_hall_call(req_id, call_origin, call_dest, call_direction)
                self.stats.log_request_assignment(req_id, current_time_step)
                if call_data in self.pending_hall_calls: self.pending_hall_calls.remove(call_data)

    def simulate_step(self, current_time_step, verbose=True):
        if verbose:
            g_status = 'ON' if G_RETURN_ENABLED else 'OFF'
            o_status = 'ON' if OPPORTUNISTIC_PICKUP_ENABLED else 'OFF'
            print(
                f"\n--- Sim Time: {current_time_step // 60:02d}:{current_time_step % 60:02d} (Step: {current_time_step}) G-Ret:{g_status} Opp:{o_status} ---")

        self.assign_requests(current_time_step)
        for elev in self.elevators: elev.step(self.stats, current_time_step)

        if verbose:
            for elev in self.elevators: print(elev)  
            g_elevators_idle = [e.id for e in self.elevators if
                                e.is_effectively_idle() and e.current_floor == GROUND_FLOOR]
            print(f"Elevators idle at G: {g_elevators_idle if g_elevators_idle else 'None'}")
            print(f"Pending Hall Calls: {len(self.pending_hall_calls)}")


def get_current_request_probability(t):  
    if MORNING_PEAK_START <= t < MORNING_PEAK_END or EVENING_PEAK_START <= t < EVENING_PEAK_END:
        return PROB_PEAK_TRAFFIC
    elif LOW_TRAFFIC_START <= t < LOW_TRAFFIC_END:
        return PROB_LOW_TRAFFIC
    else:
        return PROB_NORMAL_TRAFFIC


def get_request_type_bias(t):  
    if MORNING_PEAK_START <= t < MORNING_PEAK_END:
        return "TO_GROUND"
    elif EVENING_PEAK_START <= t < EVENING_PEAK_END:
        return "FROM_GROUND"
    else:
        return "BALANCED"


def run_simulation(g_return_flag, opportunistic_flag, max_sim_time_minutes, verbose_output=True,
                   random_seed=None):  
    global G_RETURN_ENABLED, OPPORTUNISTIC_PICKUP_ENABLED
    G_RETURN_ENABLED = g_return_flag
    OPPORTUNISTIC_PICKUP_ENABLED = opportunistic_flag
    if random_seed is not None: random.seed(random_seed)
    print(
        f"\n--- Starting 24-Hour Sim: G-Ret:{'ON' if G_RETURN_ENABLED else 'OFF'}, Opp:{'ON' if OPPORTUNISTIC_PICKUP_ENABLED else 'OFF'} ---")
    building = Building();
    request_id_counter = 0
    for time_step in range(max_sim_time_minutes):
        current_prob = get_current_request_probability(time_step)
        request_bias = get_request_type_bias(time_step)
        if random.random() < current_prob:
            request_id_counter += 1;
            building.total_requests_generated_this_simulation += 1
            start_floor, dest_floor, call_direction = -1, -1, None
            is_to_ground = (request_bias == "TO_GROUND" and random.random() < 0.7) or \
                           (request_bias == "FROM_GROUND" and random.random() < 0.3) or \
                           (request_bias == "BALANCED" and random.random() < 0.5)
            if is_to_ground:
                start_floor = random.randint(1, NUM_FLOORS - 1);dest_floor = GROUND_FLOOR;call_direction = "DOWN"
            else:
                start_floor = GROUND_FLOOR;dest_floor = random.randint(1, NUM_FLOORS - 1);call_direction = "UP"
            if start_floor == dest_floor:  
                if start_floor == GROUND_FLOOR:
                    dest_floor = 1; call_direction = "UP"
                else:
                    dest_floor = GROUND_FLOOR; call_direction = "DOWN"
            building.add_request(start_floor, dest_floor, call_direction, time_step, request_id_counter)
        building.simulate_step(time_step, verbose=verbose_output)
        if verbose_output and time_step % 60 == 0: time.sleep(0.0001)  
    if verbose_output: print(f"\n--- SIMULATION ENDED ({max_sim_time_minutes // 60} hours) ---")
    print(
        f"\n--- FINAL 24-HOUR SIMULATION STATS (G-Ret:{'ON' if G_RETURN_ENABLED else 'OFF'}, Opp:{'ON' if OPPORTUNISTIC_PICKUP_ENABLED else 'OFF'}) ---")
    print(f"Total sim time: {max_sim_time_minutes // 60}h ({max_sim_time_minutes}m)")
    print(f"Total requests generated: {building.total_requests_generated_this_simulation}")
    print(f"Total requests completed: {building.stats.requests_completed}")
    uncompleted = building.total_requests_generated_this_simulation - building.stats.requests_completed
    if uncompleted > 0: print(f"Requests pending at end: {uncompleted}")
    print(f"Overall Avg wait: {building.stats.get_average_wait_time():.2f}m")
    print(f"Overall Max wait: {building.stats.max_wait_time}m")
    print(f"Overall Avg trip: {building.stats.get_average_trip_time():.2f}m")
    building.stats.print_period_stats()
    print("\nElevator Stats:")
    for elev in building.elevators:
        util = (elev.non_idle_time_steps / max_sim_time_minutes) * 100 if max_sim_time_minutes > 0 else 0
        g_info = f"(Desig.G-Ret, Empty G-Trips: {elev.empty_g_return_trips_initiated})" if elev.is_designated_g_return_elevator else ""
        print(
            f"  {elev.id}{g_info}: Util: {util:.2f}% ({elev.non_idle_time_steps}/{max_sim_time_minutes} non-idle min)")
    return {}


if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="Simulate 24-hour elevator traffic with advanced options.")
    parser.add_argument("-G", "--g_return", action="store_true", help="Enable G-floor return logic for E1.")
    parser.add_argument("-N", "--no_g_return", action="store_false", dest="g_return",
                        help="Disable G-floor return logic (default).")
    parser.set_defaults(g_return=False)
    parser.add_argument("-O", "--opportunistic_pickup", action="store_true",
                        help="Enable opportunistic (en-route) pickups.")
    parser.add_argument("-noO", "--no_opportunistic_pickup", action="store_false", dest="opportunistic_pickup",
                        help="Disable opportunistic pickups (default).")
    parser.set_defaults(opportunistic_pickup=False)
    parser.add_argument("-s", "--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress verbose step-by-step output.")
    args = parser.parse_args();
    sim_minutes = 24 * 60
    run_simulation(
        g_return_flag=args.g_return, opportunistic_flag=args.opportunistic_pickup,
        max_sim_time_minutes=sim_minutes, verbose_output=not args.quiet, random_seed=args.seed
    )