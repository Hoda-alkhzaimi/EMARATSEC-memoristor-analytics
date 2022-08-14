function val  = val2de(x)
s = sprintf('%.10e',x);

    pos = strfind(s, 'e');

    ex = str2double(s(pos+2:end));

    tot = sprintf('%.9f', x*10^ex);
    val = (str2double(tot(strfind(tot,'.')+1:end)));
    %val2 = dec2bin(val);

end