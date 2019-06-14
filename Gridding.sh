#!/usr/bin/env bash

species=${1}
if [[ $# -lt 1 ]]
    then echo "usage: ${0} <species>"
    echo "species can be one of NO2, O3, CO or SO2"
    exit
fi

#set -x
basedir='/lustre/storeB/project/fou/kl/vals5p/'
if [[ ${species} == 'NO2' ]]
    then
    dldir="${basedir}download/tar/"
    dldir="${basedir}download/tar/2018/02/"
    dlext=".tar"
    varname='nitrogendioxide_tropospheric_column'
else
    echo "species can be one of NO2, O3, CO or SO2"
    exit 1
fi

L2dir="${basedir}data/daily_L2/"
L3dir="${basedir}data/daily_L3/"
downloaddir="${dldir}"
#netcdfdir="${datadir}netcdf/"
#netcdfdir="${datadir}netcdf_himalaya_domain_${species}/"
#modeloutdir="${basedir}EMEPmodel.colocated.${retrieval}/"

jobfile="./S5P.run.${species}.gridding.txt"
rm -f "${jobfile}"

#mkdir -p "${netcdfdir}"
for file in `find ${downloaddir} -name "*${dlext}" | sort`
    do echo ${file}
    if [[ ${species} == 'NO2' ]]
        then date=`basename ${file} ${dlext} | cut -d_ -f3`
    else
        echo 'not yet implemented'
    fi
    griddedfile="${L3dir}/gridded_${date}.nc"
    ungriddedfile="${L2dir}/ungridded_${date}.nc"
    plotfile="${L3dir}/gridded_${date}.png"
#    cmd="./simplegridder.py --retrieval ${species}  --outdir ${netcdfdir} --plotmap --tempdir /tmp/ --file ${file}"
#    -O --gridfile ./gridded.nc --variables nitrogendioxide_tropospheric_column --outfile ./ungridded.nc --file /home/jang/tmp/tropomi_no2_20181201.tar
    cmd="./simplegridder.py -O --gridfile ${griddedfile} --variables ${varname}  --outfile ${ungriddedfile} --plotmap --plotdir ${L3dir} --file ${file}"
    echo ${cmd}
    echo ${cmd} >> "${jobfile}"
done
exit

# start using gnu parallel
/usr/bin/parallel -vk -j 5 -a "${jobfile}"
