# -*- coding: utf-8 -*-
"""
Library of functions for reading Kuna machine .d format. Requires package binio

Created on Tue Aug  5 16:36:03 2014

@author: jan_cimbalnik
"""

import binio,os
import numpy as np

def read_d_header(file_name,read_tags=False):
    """
    Function to read .d file standard and extended header

    Parameters:
    ----------
        file_name - name of the file (string)\n
        read_tags - whether the tags should be read(default=False)\n

    Returns:
    ----------
        sheader - standard header (dictionary)\n
        xheader - extended header (dictionary)\n
    """
    # Open the .d file
    f =  open(file_name,"rb")

    # Reading standard header

    sheader_struct = binio.new (
        "# Standard header struct\n"
        "15  : string    : sign\n"
        "1   : string    : ftype\n"
        "1   : uint8     : nchan\n"
        "1   : uint8     : naux\n"
        "1   : uint16    : fsamp\n"
        "1   : uint32    : nsamp\n"
        "1   : uint8     : d_val # This is to be done\n"
        "1   : uint8     : unit\n"
        "1   : uint16    : zero\n"
        "1   : uint16    : data_org\n"
        "1   : int16     : data_xhdr_org\n"
    )

    sheader = sheader_struct.read_dict(f)
    sheader['file_name'] = file_name
    # Create a dictionary in d_val field - to be done
    dval_dict={}
    dval_dict['value']=sheader['d_val']
    dval_dict['data_invalid']=int(np.floor(sheader['d_val']/128)%2)
    dval_dict['data_packed']=int(np.floor(sheader['d_val']/64)%2)
    dval_dict['block_structure']=int(np.floor(sheader['d_val']/32)%2)
    dval_dict['polarity']=int(np.floor(sheader['d_val']/16)%2)
    dval_dict['data_calib']=int(np.floor(sheader['d_val']/8)%2)
    dval_dict['data_modified']=int(np.floor(sheader['d_val']/4)%2)
    dval_dict['data_cell_size']=int(sheader['d_val']%4)
    sheader['d_val']=dval_dict


    # Fix the data_org_fields
    sheader['data_org'] = 16*sheader['data_org']
    sheader['data_xhdr_org'] = 16*sheader['data_xhdr_org']



    ### Reading extended header
    xheader={}
    # We need to define a dictionary for xhdr (we don't have switch-case)
    xhdr_dict={ 16725: [" : uint8 : authentication_key\n",0],
                22082: [" : uint8 : block_var_list\n",0],
                16707: [" : uint8 : channel_atrib\n",0],
                18755: [" : uint8 : calib_info\n",0],
                20035: [" : uint8 : channel_names\n",0], #Finish here
                17988: [" : uint8 : dispose_flags\n",0],
                18756: [" : char  : data_info\n",0],
                19526: [" : char  : file_links\n",0],
                21318: [" : int16 : freq\n",2], # Finish here
                17481: [" : uint32: patient_id\n",1], # Finish here
                19024: [" : char  : project_name\n",0],
                16978: [" : uint8 : rblock\n",0],
                18003: [" : char  : source_file\n",0],
                17748: [" : char  : text_record\n",0],
                18772: [" : uint32: time_info\n",1], # UTC from 1.1.1970
                21588: [" : uint16: tag_tableen\n",2, " : uint32: tag_tableoff\n",2],
                22612: [" : char  : text_extrec\n",0],
                0    : 0
                }

    if sheader['data_xhdr_org']!=0:
        f.seek(sheader['data_xhdr_org'])
        cont = 1
        while cont:
            field_struct = binio.new(
            "1   : uint16    : mnemo\n"
            "1   : uint16    : field_len\n"
            )
            field = field_struct.read_dict(f)

            #xhdr dictionary check
            if field['mnemo'] in xhdr_dict:

                if field['mnemo']==0:
                    cont = 0
                    sheader['datapos']=f.tell()
                elif field['mnemo']==21588:
                    io_txt=str(xhdr_dict[field['mnemo']][1])+xhdr_dict[field['mnemo']][0]+str(xhdr_dict[field['mnemo']][3])+xhdr_dict[field['mnemo']][2]
                    field_cont = binio.new(io_txt)
                    xheader.update(field_cont.read_dict(f))
                elif xhdr_dict[field['mnemo']][1]==0:
                    io_txt=str(field['field_len'])+xhdr_dict[field['mnemo']][0]
                    field_cont = binio.new(io_txt)
                    xheader.update(field_cont.read_dict(f))
                else:
                    io_txt=str(xhdr_dict[field['mnemo']][1])+xhdr_dict[field['mnemo']][0]
                    field_cont = binio.new(io_txt)
                    xheader.update(field_cont.read_dict(f))

            else:
                f.seek(field['field_len'],1)

    #Now fix stuff

    if 'channel_names' in xheader:
        char_lst = [chr(x) for x in xheader['channel_names']]
        channels = [''.join(char_lst[x:x+4]) for x in range(0,len(char_lst),4)]
        channels = [x.replace('\x00','') for x in channels]
        xheader['channel_names'] = channels
    if 'tag_tableen' in xheader:
        xheader['tag_table']=xheader['tag_tableen']+xheader['tag_tableoff']
        xheader.pop('tag_tableen')
        xheader.pop('tag_tableoff')

    ### Read tags
    if 'tag_table' in xheader and read_tags:
        xheader['tags'] = {}
        xheader['tags']['tag_pos'] = []
        xheader['tags']['tag_class'] = []
        xheader['tags']['tag_selected'] = []
        xheader['tags']['classes'] = []
        #Fix datasets bug in D-file with defoff and listoff
        pr,nb = get_prec(sheader)
        while xheader['tag_table'][2] < (sheader['nchan']*sheader['nsamp']*nb + sheader['data_org']) and xheader['tag_table'][2] > sheader['datapos']:
            xheader['tag_table'][2] += 16 ** 8
            xheader['tag_table'][3] += 16 ** 8

        #Read tag list
        f.seek(xheader['tag_table'][3])
        tag_list_struct = binio.new(
        str(int(4 * np.floor(float(xheader['tag_table'][1])/4.0)))+"   : uint8    : tlist")
        tag_list = tag_list_struct.read_dict(f)
        tag_list = tag_list['tlist']
        tag_list = [tag_list[x:x+4] for x in range(0,len(tag_list),4)]
        for x in range(len(tag_list)):
            tag_list[x][2] = tag_list[x][2] % 128
        tag_pos = []
        for x in tag_list:
            tag_pos.append((x[0]*1) + (x[1]*256) + (x[2]*65536))
        xheader['tags']['tag_pos'] = tag_pos
        xheader['tags']['tag_class'] = [x[3] % 128 for x in tag_list]
        xheader['tags']['tag_selected'] = [np.floor(float(x[3])/128.0) for x in tag_list]

        #Fix the long datasets bug in D-file (positions > 2 ** 23-1)
        if xheader['tags']['tag_pos'] != []:
            cont1 = 1
            while cont1:
                wh = [list(np.diff(xheader['tags']['tag_pos'])).index(x) for x in list(np.diff(xheader['tags']['tag_pos'])) if x < 0]
                if wh == []:
                    cont1 = 0
                else:
                    xheader['tags']['tag_pos'][wh[0]+1:] = [x + (2 ** 23) for x in xheader['tags']['tag_pos'][wh[0]+1:]]

        #Read the tag table
        currpos = xheader['tag_table'][2]
        cont1 = 1
        while cont1:
            f.seek(currpos)
            tag_read_dic = binio.new(
            "2  :   char    : abrv\n"
            "1  :   uint16  : n\n"
            "1  :   uint16  : txtlen\n"
            "1  :   uint16  : txtoff\n")

            curr_tag = tag_read_dic.read_dict(f)
            currpos += 8
            if np.floor(curr_tag['n']/32768):
                cont1 = 0
            f.seek(curr_tag['txtoff']+xheader['tag_table'][2])
            xheader['tags']['classes'].append([curr_tag['abrv'],curr_tag['n'] % 32768,str(f.read(curr_tag['txtlen']))])

    return sheader,xheader

def read_d_data(sheader,ch_list,samp_start,samp_end):
    """
    Function to read the data from .d file

    Parameters:
    ----------

    sheader - standard header (dictionary)\n
    ch_list - list of channel idxs (list)\n
    samp_start - start sample (int)\n
    samp_end - end sample (int)\n

    Returns:
    --------
    data - data from .d file (list), corresponding to ch_list\n

    """

    ### Open the .d file
    f = open (sheader['file_name'],"rb")

    ### Retype the start and end
    samp_start = int(samp_start)
    samp_end = int(samp_end)

    ### Just a precaution
    if samp_end >= sheader['nsamp']:
        print("\n End sample "+str(samp_end)+" equal or bigger than n samples of file ("+str(sheader['nsamp'])+"), setting to the end of the file \n")
        samp_end = sheader['nsamp']-1

    ### Get the precision - we have function for this!!!
    prec,nb = get_prec(sheader)

    ### Get the first sample
    ds1 = sheader['data_org'] + (samp_start-1)*nb*sheader['nchan']
    n_samp = samp_end-samp_start
    f.seek(ds1)

    ### Read the data
    dat_type=binio.new(str(n_samp*sheader['nchan'])+" : "+prec+" : data")
    try:
        f.seek(ds1)
        dd = dat_type.read_struct(f)
    except:
        print ('Error reading data')
        #print '\n '+str(samp_end)+' in '+str(sheader['nsamp'])+' \n'
        return

    ### Now get the desired channel
#    data = np.zeros([len(ch_list),n_samp],dtype=prec)
#    for i,x in enumerate(ch_list):
#        data[i] = np.array(dd.data[x::sheader['nchan']])

    return np.array(dd.data).reshape(n_samp,sheader['nchan'])[:,ch_list]

def get_prec(sheader):
    """
    Helper funciton for reading tags

    Parameters:
    ----------
    sheader\n

    Returns:
    --------
    precision,nb
    """
    if sheader['ftype'] == u'D':
        if sheader['d_val']['data_cell_size'] == 2:
            return 'int16',2
        elif sheader['d_val']['data_cell_size'] == 3:
            return 'int32',4
        else:
            return 'uint8',1
    elif sheader['ftype'] == u'R':
        return 'float32',4
    else:
        return 'uint8',1

#def d2gdf:












