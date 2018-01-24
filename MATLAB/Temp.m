clear all; close all; clc;
HebiKeyboard.loadLibs();
%%
kb = HebiKeyboard();
while true
    state = read(kb);
    if all(state.keys('w'))
        disp('w is both pressed!')
    end
    if all(state.keys('a'))
        disp('a is both pressed!')
    end
    if all(state.keys('s'))
        disp('s is both pressed!')
    end
    if all(state.keys('d'))
        disp('d is both pressed!')
    end
    pause(0.05);
end
%%
kb = HebiKeyboard();
while true
    state = read(kb);
    down = find(state.keys('a':'z')) + 'a';
    if ~state.SHIFT
        disp(char(down));
    end
    pause(0.01);
end
%%
joy = HebiJoystick(1);
while true
  [axes, buttons, povs] = read(joy);
  if any(buttons)
    disp(['Pressed buttons: ' num2str(find(buttons))]);
  end
  pause(0.1);
end