clc;
clear vars;
clear all;
Nb = 4
ishealthy = true;
n_bars = 1:1:3;
Rs = 11.2; %stator resistance
Rr = 8.3; %rotor resistance
Ls = 0.6155; %stator inductanec
Lr = 0.6380; %rotor inductance
Lm = 0.57; %mutual inductance
J = 0.00214; %inertia
B = 0.0041; %
poles = [2,4]; %pole pairs
f = 50; %frequency
we = 2*pi*f;
if ishealthy=true
        n = 0
        Tl = 1
        while Tl< 6
            Rbrenew = ((3*Rbr)/(Nb-n))-((3*Rbr)/Nb);
            Rr = Rr + Rbrenew;
        
            sim_model = sim("detailed_model.slx");
                            
            rotorcurrents = get(sim_model, 'rotorcurrents');
            statorcurrents = get(sim_model, 'statorcurrents');
            voltage = get(sim_model, 'voltage');
            speeds = get(sim_model, 'speed');
            torques = get(sim_model,'torque')
            stator_thd = get(sim_model,'statorthd')
            rotor_a = rotorcurrents(end,1);
            rotor_b = rotorcurrents(end,2);
            rotor_c = rotorcurrents(end,3);
            stator_a = statorcurrents(end,1);
            stator_b = statorcurrents(end,2);
            stator_c = statorcurrents(end,3);
            voltage_thd = voltage(end,:)
            speed = speeds(end,1);
            torque = torques(end,1);
            time = sim_model.tout(end,:)
            concat = horzcat(stator_a, stator_b, stator_c, rotor_a, rotor_b, rotor_c,torque,speed,p,voltage_thd,stator_thd);
            filename = strcat('Data/',num2str(n_bars), '/',num2str(Tl),'/','bars_data.csv')
        end
end

if ishealthy == false
    for tr = 1:length(n_bars)
        n = n_bars(tr);
        Tl = 1
        while Tl< 6
            Rbrenew = ((3*Rbr)/(Nb-n))-((3*Rbr)/Nb);
            Rr = Rr + Rbrenew;
        
            sim_model = sim("detailed_model.slx");
                            
            rotorcurrents = get(sim_model, 'rotorcurrents');
            statorcurrents = get(sim_model, 'statorcurrents');
            voltage = get(sim_model, 'voltage');
            speeds = get(sim_model, 'speed');
            torques = get(sim_model,'torque')
            stator_thd = get(sim_model,'statorthd')
            rotor_a = rotorcurrents(end,1);
            rotor_b = rotorcurrents(end,2);
            rotor_c = rotorcurrents(end,3);
            stator_a = statorcurrents(end,1);
            stator_b = statorcurrents(end,2);
            stator_c = statorcurrents(end,3);
            voltage_thd = voltage(end,:)
            speed = speeds(end,1);
            torque = torques(end,1);
            time = sim_model.tout(end,:)
            concat = horzcat(stator_a, stator_b, stator_c, rotor_a, rotor_b, rotor_c,torque,speed,p,voltage_thd,stator_thd);
            filename = strcat('Data/',num2str(n_bars),'/',num2str(Tl),'/','bars_data.csv')
            Tl = Tl + 0.1
        end
    end
end
