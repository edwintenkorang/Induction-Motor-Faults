
 f = 50;
torque_full = 26.9;
loads = 0.1:0.1:1;
times = 0.3:0.1:0.8;
status = 'ABCG';
for tr = 1:length(loads)
        load = loads(tr);
        for time = 1:length(times)
            t = times(time);
       
                torque = load*torque_full;
                sim_model = sim("motorfault.slx");
                
                phaseacurrent = get(sim_model, 'statorcurrenta');
                phasebcurrent = get(sim_model, 'statorcurrentb');
                phaseccurrent = get(sim_model, 'statorcurrentc');
                newtorque = get(sim_model,'outtorque');
                newspeed = get(sim_model,'speed');
                [c,l] = wavedec(phaseacurrent, 9, 'db4');
                [Ea1,Ed1] = wenergy(c,l);
                [c,l] = wavedec(phasebcurrent, 9, 'db4');
                [Ea2,Ed2] = wenergy(c,l);
                [c,l] = wavedec(phaseccurrent, 9, 'db4');
                [Ea3,Ed3] = wenergy(c,l);
                [c,l] = wavedec(newtorque, 9, 'db4');
                [Ea4,Ed4] = wenergy(c,l);
                [c,l] = wavedec(newspeed, 9, 'db4');
                [Ea5,Ed5] = wenergy(c,l);        
                data1 = horzcat(Ed1, Ed2, Ed3,Ed4,Ed5);
                data1(1,46) = load;
                foldername = strcat('Data\',status,'\',num2str(load),'\',num2str(t),'\');
                mkdir(foldername);
                filename = strcat(foldername,'matrix.csv');
        
                csvwrite(filename, data1);
        end


end