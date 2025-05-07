import random
import time
import argparse
from collections import defaultdict

NUM_ELEVATORS = 3
NUM_FLOORS = 29 # Ground floor (0) + 29 units (1-29)
GROUND_FLOOR = 0
MAX_CAPACITY = 8
DOOR_CYCLE_TIME = 1  # Time steps for door to open and close
ELEVATOR_SPEED = 1  # Time steps to move one floor

# Population and Commute Parameters
POPULATION_PER_FLOOR = 2
# TOTAL_POPULATION = NUM_FLOORS * POPULATION_PER_FLOOR
MORNING_RUSH_START_MINUTES = 7 * 60
MORNING_RUSH_END_MINUTES = 9 * 60
EVENING_RUSH_START_MINUTES = 17 * 60
EVENING_RUSH_END_MINUTES = 19 * 60
COMMUTER_PERCENTAGE = 0.80 # 80% of residents commute
PROB_MISC_TRIP_PER_PERSON_PER_HOUR = 0.05 # Chance per person per hour of making a non-commute trip


G_RETURN_ENABLED = False
OPPORTUNISTIC_PICKUP_ENABLED = False

class Elevator:
    def __init__(self, id_num, is_designated_g_return_elevator=False):
        self.id = f"E{id_num}"
        self.current_floor = GROUND_FLOOR
        self.state = "IDLE"
        self.is_designated_g_return_elevator = is_designated_g_return_elevator
        self.time_in_current_state = 0
        self.passengers_inside = {}
        self.committed_pickups = {}
        self.current_direction = None
        self.next_target_stop = None
        self.non_idle_time_steps = 0
        self.empty_g_return_trips_initiated = 0

    def __repr__(self):
        g_ret_marker = f"(G-Ret:{'ON' if self.is_designated_g_return_elevator and G_RETURN_ENABLED else 'OFF'})" 
        opp_marker = f"(Opp:{'ON' if OPPORTUNISTIC_PICKUP_ENABLED else 'OFF'})" 
        pass_dests = sorted([p['dest'] for p in self.passengers_inside.values()])
        pickup_origins = sorted([p['origin'] for p in self.committed_pickups.values()])
        return (f"{self.id} " 
                f"F:{self.current_floor} S:{self.state} Dir:{self.current_direction} Target:{self.next_target_stop} "
                f"PassDests:{pass_dests} PickupOrigins:{pickup_origins}")

    def is_idle(self):
        return self.state == "IDLE" and not self.passengers_inside and not self.committed_pickups

    def has_tasks(self):
        return bool(self.passengers_inside or self.committed_pickups)

    def add_assigned_hall_call(self, req_id, origin, dest, direction):
        self.committed_pickups[req_id] = {'origin': origin, 'dest': dest, 'direction': direction}
        if self.is_idle():  
            self._determine_next_target_and_direction()

    def _determine_next_target_and_direction(self):
        if not self.has_tasks():
            self.current_direction = None
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
        determined_direction = self.current_direction  

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
                                determined_direction = self.current_direction  

        self.next_target_stop = determined_target
        self.current_direction = determined_direction if determined_target is not None else None

        if self.next_target_stop is not None:
            if self.current_floor == self.next_target_stop:
                self.state = "DOOR_CYCLE"; self.time_in_current_state = 0
            elif self.next_target_stop > self.current_floor:
                self.current_floor += 1; self.current_direction = "UP"
            elif self.next_target_stop < self.current_floor:
                self.current_floor -= 1; self.current_direction = "DOWN"


    def step(self, building_stats_collector, current_sim_time):
        if self.state != "IDLE" or self.has_tasks(): self.non_idle_time_steps += 1
        self.time_in_current_state += 1

        if self.state == "IDLE":
            if not self.has_tasks():
                if G_RETURN_ENABLED and self.is_designated_g_return_elevator and self.current_floor != GROUND_FLOOR:
                    self.current_direction = "DOWN" if self.current_floor > GROUND_FLOOR else "UP"
                    if self.current_floor == GROUND_FLOOR:
                        self.current_direction = None; self.state = "IDLE"
                    else:
                        self.next_target_stop = GROUND_FLOOR; self.state = "MOVING"
                    self.empty_g_return_trips_initiated += 1
            else:
                self._determine_next_target_and_direction()
            return

        elif self.state == "DOOR_CYCLE":
            if self.time_in_current_state >= DOOR_CYCLE_TIME:
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
                        if self.current_direction is None or c[
                            'direction'] == self.current_direction or not self.passengers_inside:
                            if self.current_direction is None and not self.passengers_inside: self.current_direction = \
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
                self.current_floor += 1; self.current_direction = "UP"
            elif self.next_target_stop < self.current_floor:
                self.current_floor -= 1; self.current_direction = "DOWN"


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
        if MORNING_RUSH_START_MINUTES <= t < MORNING_RUSH_END_MINUTES or EVENING_RUSH_START_MINUTES <= t < EVENING_RUSH_END_MINUTES:
            return "peak"
        elif 0 <= t < MORNING_RUSH_START_MINUTES:
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

    def get_average_trip_time(self, period=None):  # Not used with new request model
        # The 'period' argument is no longer used for the overall average.
        # If period-specific averages were needed, logic would be re-added here.
        if self.requests_completed > 0:
            return self.total_trip_time / self.requests_completed
        return 0

    def get_average_wait_time(self, p=None):
        if p: ts = self.wait_times_by_period[p]; return sum(ts) / len(ts) if ts else 0
        num_assigned = len(self.request_assignment_times)
        return self.total_wait_time / num_assigned if num_assigned > 0 else 0


    def get_period_stats_lines(self):
        lines = ["Performance by Traffic Period: (N/A for current population model)"]
        # for p in ["low", "normal", "peak"]:\n        #     req_count = len(self.wait_times_by_period[p])\n        #     avg_w = self.get_average_wait_time(p)\n        #     avg_t = self.get_average_trip_time(p)\n        #     lines.append(f"  {p.capitalize()} Traffic ({req_count} reqs): Avg Wait: {avg_w:.2f} min, Avg Trip: {avg_t:.2f} min")
        return lines


class Building:
    def __init__(self, num_elevators=NUM_ELEVATORS, num_floors=NUM_FLOORS):
        self.elevators = [Elevator(i + 1, (i == 0)) for i in range(num_elevators)]
        self.pending_hall_calls = []
        self.stats = StatsCollector()
        self.total_requests_generated_this_simulation = 0

    def add_request(self, start_floor, dest_floor, direction, current_time_step, req_id):
        self.pending_hall_calls.append(
            {'origin': start_floor, 'dest': dest_floor, 'direction': direction, 'req_id': req_id,
             'time': current_time_step})
        self.stats.log_request_creation(req_id, current_time_step)
        self.total_requests_generated_this_simulation += 1

    def assign_requests(self, current_time_step, verbose=False, log_stream=None):
        calls_processed_this_step = []

        def _log_assign(message):
            if verbose:
                print(message)
            if log_stream:
                log_stream.write(message + "\n")

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
                    if not elev.current_direction or elev.current_direction == call_direction:
                        can_pickup = False
                        if call_direction == "UP" and elev.current_floor <= call_origin:
                            can_pickup = True
                        elif call_direction == "DOWN" and elev.current_floor >= call_origin:
                            can_pickup = True

                        if elev.is_idle() and not elev.has_tasks():
                            can_pickup = True

                        if can_pickup:
                            dist = abs(elev.current_floor - call_origin)
                            if dist < min_dist_to_pickup:
                                min_dist_to_pickup = dist
                                best_opportunistic_elevator = elev

                if best_opportunistic_elevator:
                    _log_assign(f"OPPORTUNISTIC ASSIGN: Call ID {req_id} (F{call_origin}->F{call_dest} Dir:{call_direction}) to Elevator {best_opportunistic_elevator.id}")
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

            idle_candidates = [e for e in self.elevators if e.is_idle()]
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
                _log_assign(f"STANDARD ASSIGN: Call ID {req_id} (F{call_origin}->F{call_dest} Dir:{call_direction}) to Elevator {best_standard_elevator.id}")
                best_standard_elevator.add_assigned_hall_call(req_id, call_origin, call_dest, call_direction)
                self.stats.log_request_assignment(req_id, current_time_step)
                if call_data in self.pending_hall_calls: self.pending_hall_calls.remove(call_data)

    def simulate_step(self, current_time_step, verbose=True, log_stream=None):
        def _log_sim_step(message):
            if verbose:
                print(message)
            if log_stream:
                log_stream.write(message + "\n")

        # Perform actions first
        self.assign_requests(current_time_step, verbose=verbose, log_stream=log_stream) # verbose here for its own logs
        for elev in self.elevators: elev.step(self.stats, current_time_step)

        if verbose: # Now log based on the state AFTER actions
            any_elevator_active = any(not e.is_idle() for e in self.elevators)
            no_pending_calls = not self.pending_hall_calls

            if (not any_elevator_active) and no_pending_calls:
                time_str = f"{current_time_step // 60:02d}:{current_time_step % 60:02d}"
                g_status = 'ON' if G_RETURN_ENABLED else 'OFF'
                o_status = 'ON' if OPPORTUNISTIC_PICKUP_ENABLED else 'OFF'
                idle_positions_str = ", ".join([f"E{e.id}:F{e.current_floor}" for e in self.elevators])
                _log_sim_step(f"{time_str} (G-Ret:{g_status} Opp:{o_status}) - All Idle & No Calls - Positions: {idle_positions_str}")
            else:
                g_status = 'ON' if G_RETURN_ENABLED else 'OFF'
                o_status = 'ON' if OPPORTUNISTIC_PICKUP_ENABLED else 'OFF'
                header_message = f"\n--- Sim Time: {current_time_step // 60:02d}:{current_time_step % 60:02d} (Step: {current_time_step}) G-Ret:{g_status} Opp:{o_status} ---"
                _log_sim_step(header_message)

                if any_elevator_active:
                    for elev in self.elevators:
                        _log_sim_step(str(elev))
                else: # All elevators are idle, but there must be pending calls
                    idle_positions_str = ", ".join([f"E{e.id}:F{e.current_floor}" for e in self.elevators])
                    _log_sim_step(f"All elevators idle. Positions: {idle_positions_str}")

                g_elevators_idle = [e.id for e in self.elevators if e.is_idle() and e.current_floor == GROUND_FLOOR]
                _log_sim_step(f"Elevators idle at G: {g_elevators_idle if g_elevators_idle else 'None'}")
                _log_sim_step(f"Pending Hall Calls: {len(self.pending_hall_calls)}")


def run_simulation(g_return_flag, opportunistic_flag, max_sim_time_minutes, verbose_output=True,
                   random_seed=None, log_stream=None):
    global G_RETURN_ENABLED, OPPORTUNISTIC_PICKUP_ENABLED
    G_RETURN_ENABLED = g_return_flag
    OPPORTUNISTIC_PICKUP_ENABLED = opportunistic_flag

    if random_seed is not None: random.seed(random_seed)

    def _log(message):
        if verbose_output:
            print(message)
        if log_stream:
            log_stream.write(message + "\n")

    _log(f"SIMULATION INIT: Max Time: {max_sim_time_minutes}m, G-Return: {G_RETURN_ENABLED}, Opportunistic: {OPPORTUNISTIC_PICKUP_ENABLED}, Seed: {random_seed}, Verbose: {verbose_output}")

    total_population = (NUM_FLOORS -1) * POPULATION_PER_FLOOR # Exclude ground floor for units
    residents = [] # List of {'id': i, 'home_floor': unit_floor}
    current_floor_assignment = 1
    for i in range(total_population):
        residents.append({'id': i, 'home_floor': current_floor_assignment})
        if (i+1) % POPULATION_PER_FLOOR == 0:
            current_floor_assignment +=1
            if current_floor_assignment > NUM_FLOORS: # Should not happen if NUM_FLOORS is units+G
                 current_floor_assignment = 1 # Reset, though ideally NUM_FLOORS is total including G

    _log(f"SIMULATION INIT: Floors: {NUM_FLOORS}, Pop/Floor: {POPULATION_PER_FLOOR}, Total Pop: {total_population}")
    _log(f"Params: Max Time: {max_sim_time_minutes}m, G-Return: {G_RETURN_ENABLED}, Opportunistic: {OPPORTUNISTIC_PICKUP_ENABLED}, Seed: {random_seed}, Verbose: {verbose_output}")

    building = Building(NUM_ELEVATORS, NUM_FLOORS) # Pass correct NUM_FLOORS
    request_id_counter = 0

    pre_generated_requests = []
    num_commuters = int(total_population * COMMUTER_PERCENTAGE)
    commuter_ids = random.sample(range(total_population), num_commuters)

    for resident_id in commuter_ids:
        resident_home_floor = residents[resident_id]['home_floor']

        # Morning Departure
        morning_departure_time = random.randint(MORNING_RUSH_START_MINUTES, MORNING_RUSH_END_MINUTES)
        pre_generated_requests.append({
            'time': morning_departure_time, 'id': f"comm_dep_{resident_id}",
            'start': resident_home_floor, 'dest': GROUND_FLOOR, 'direction': "DOWN"
        })

        # Evening Return
        evening_return_time = random.randint(EVENING_RUSH_START_MINUTES, EVENING_RUSH_END_MINUTES)
        pre_generated_requests.append({
            'time': evening_return_time, 'id': f"comm_ret_{resident_id}",
            'start': GROUND_FLOOR, 'dest': resident_home_floor, 'direction': "UP"
        })

    pre_generated_requests.sort(key=lambda r: r['time'])

    _log(f"Pre-generated {len(pre_generated_requests)} commute requests for {num_commuters} commuters.")

    for time_step in range(max_sim_time_minutes):
        # Process pre-generated requests for this time_step
        while pre_generated_requests and pre_generated_requests[0]['time'] == time_step:
            req = pre_generated_requests.pop(0)
            request_id_counter += 1
            # Use descriptive req ID from pre-generation, or make a new one if needed
            actual_req_id = f"{req['id']}_{request_id_counter}"
            building.add_request(req['start'], req['dest'], req['direction'], time_step, actual_req_id)
            _log(f"PRE-GEN REQ (ID:{actual_req_id}): From F{req['start']} to F{req['dest']} (Dir:{req['direction']}) at T={time_step//60:02d}:{time_step%60:02d}")

        # Process miscellaneous requests
        prob_misc_per_min_per_person = PROB_MISC_TRIP_PER_PERSON_PER_HOUR / 60.0
        for resident_idx, resident in enumerate(residents):
            if random.random() < prob_misc_per_min_per_person:
                request_id_counter += 1
                misc_req_id = f"misc_{resident_idx}_{request_id_counter}"

                # Decide type of misc trip
                trip_type = random.choice(["unit_to_ground", "ground_to_unit"])
                start_floor, dest_floor, direction = -1, -1, None

                resident_home = resident['home_floor']

                if trip_type == "unit_to_ground":
                    start_floor = resident_home
                    dest_floor = GROUND_FLOOR
                elif trip_type == "ground_to_unit":
                    start_floor = GROUND_FLOOR
                    dest_floor = resident_home

                if start_floor == dest_floor: continue # Skip if somehow origin is destination

                if start_floor > dest_floor: direction = "DOWN"
                elif start_floor < dest_floor: direction = "UP"
                else: continue # Should not happen if start_floor != dest_floor

                building.add_request(start_floor, dest_floor, direction, time_step, misc_req_id)
                _log(f"MISC REQ (ID:{misc_req_id}): From F{start_floor} to F{dest_floor} (Dir:{direction}) at T={time_step//60:02d}:{time_step%60:02d}")

        building.simulate_step(time_step, verbose=verbose_output, log_stream=log_stream)
        if verbose_output and time_step % 15 == 0: # Reduced frequency of sleep/detailed step log
            _log(f"--- Mid-step heart beat at {time_step//60:02d}:{time_step%60:02d} ---") # Example of less frequent verbose log
            time.sleep(0.0001)

    _log(f"\n--- FINAL {max_sim_time_minutes // 60}-HOUR SIMULATION STATS (G-Ret:{'ON' if G_RETURN_ENABLED else 'OFF'}, Opp:{'ON' if OPPORTUNISTIC_PICKUP_ENABLED else 'OFF'}) ---")
    if random_seed is not None: _log(f"Seed: {random_seed}")
    _log(f"Total sim time: {max_sim_time_minutes // 60}h ({max_sim_time_minutes}m)")
    _log(f"Total requests generated: {building.total_requests_generated_this_simulation}")
    _log(f"Total requests completed: {building.stats.requests_completed}")
    uncompleted = building.total_requests_generated_this_simulation - building.stats.requests_completed
    if uncompleted > 0: _log(f"Requests pending at end: {uncompleted}")
    avg_wait = building.stats.get_average_wait_time()
    avg_trip = building.stats.get_average_trip_time()
    _log(f"Overall Avg wait: {avg_wait:.2f}m")
    _log(f"Overall Max wait: {building.stats.max_wait_time}m")
    _log(f"Overall Avg trip: {avg_trip:.2f}m")

    for line in building.stats.get_period_stats_lines():
        _log(line)

    _log("\nElevator Stats:")
    for elev in building.elevators:
        util = (elev.non_idle_time_steps / max_sim_time_minutes) * 100 if max_sim_time_minutes > 0 else 0
        g_info = f" (Desig.G-Ret, Empty G-Trips: {elev.empty_g_return_trips_initiated})" if elev.is_designated_g_return_elevator else ""
        _log(f"  {elev.id}{g_info}: Util: {util:.2f}% ({elev.non_idle_time_steps}/{max_sim_time_minutes} non-idle min)")

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
    parser.add_argument("--outfile", type=str, default=None, help="Path to a TXT file to append ALL log output and summary results.")

    args = parser.parse_args();
    sim_minutes = 24 * 60

    log_stream_main = None
    if args.outfile:
        try:
            log_stream_main = open(args.outfile, 'a')
        except IOError as e:
            print(f"Error opening output file {args.outfile}: {e}")
            log_stream_main = None

    try:
        run_simulation(
            g_return_flag=args.g_return, opportunistic_flag=args.opportunistic_pickup,
            max_sim_time_minutes=sim_minutes, verbose_output=not args.quiet, random_seed=args.seed,
            log_stream=log_stream_main
        )
    finally:
        if log_stream_main:
            log_stream_main.close()