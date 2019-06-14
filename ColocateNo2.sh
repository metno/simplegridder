#!/usr/bin/env bash

species=${1}
if [[ $# -lt 1 ]]
    then echo "usage: ${0} <species>"
    echo "species can be one of NO2, O3, CO or SO2"
    exit
fi

set -x
basedir='/lustre/storeB/project/fou/kl/vals5p/'
if [[ ${species} == 'NO2' ]]
    then
    dldir="${basdir}download/tar/"
    dlext=".tar"
else
    echo "species can be one of NO2, O3, CO or SO2"
    exit (1)
fi

L2dir=""
L3Dir=""
downloaddir="${datadir}download/AE_TD01_ALD_U_N_2A_201809*/"
#netcdfdir="${datadir}netcdf/"
netcdfdir="${datadir}netcdf_himalaya_domain_${species}/"
#modeloutdir="${basedir}EMEPmodel.colocated.${retrieval}/"

jobfile="./S5P.run.${species}.gridding.txt"
rm -f "${jobfile}"

mkdir -p "${netcdfdir}"
for file in `find ${downloaddir} -name '*.DBL' | sort`
    do echo ${file}
    # read_aeolus_l2a_data.py --himalaya --plotmap --outdir ./ --file /lustre/storeB/project/fou/kl/admaeolus/data.rev.TD01/download/AE_TD01_ALD_U_N_2A_20180908T120926033_005387992_000264_0002/AE_TD01_ALD_U_N_2A_20180908T120926033_005387992_000264_0002.DBL
#    cmd="./read_aeolus_l2a_data.py --himalaya --retrieval ${retrieval}  --outdir ${netcdfdir} --plotmap --plotprofile --tempdir /tmp/ --file ${file}"
    cmd="./read_aeolus_l2a_data.py --himalaya --retrieval ${species}  --outdir ${netcdfdir} --plotmap --tempdir /tmp/ --file ${file}"
    echo ${cmd}
    echo ${cmd} >> "${jobfile}"
done

# start using gnu parallel
/usr/bin/parallel -vk -j 5 -a "${jobfile}"
