import math


IMAGENET_CLASS_IDS = [
    "n01440764",
    "n01443537",
    "n01484850",
    "n01491361",
    "n01494475",
    "n01496331",
    "n01498041",
    "n01514668",
    "n01514859",
    "n01518878",
    "n01530575",
    "n01531178",
    "n01532829",
    "n01534433",
    "n01537544",
    "n01558993",
    "n01560419",
    "n01580077",
    "n01582220",
    "n01592084",
    "n01601694",
    "n01608432",
    "n01614925",
    "n01616318",
    "n01622779",
    "n01629819",
    "n01630670",
    "n01631663",
    "n01632458",
    "n01632777",
    "n01641577",
    "n01644373",
    "n01644900",
    "n01664065",
    "n01665541",
    "n01667114",
    "n01667778",
    "n01669191",
    "n01675722",
    "n01677366",
    "n01682714",
    "n01685808",
    "n01687978",
    "n01688243",
    "n01689811",
    "n01692333",
    "n01693334",
    "n01694178",
    "n01695060",
    "n01697457",
    "n01698640",
    "n01704323",
    "n01728572",
    "n01728920",
    "n01729322",
    "n01729977",
    "n01734418",
    "n01735189",
    "n01737021",
    "n01739381",
    "n01740131",
    "n01742172",
    "n01744401",
    "n01748264",
    "n01749939",
    "n01751748",
    "n01753488",
    "n01755581",
    "n01756291",
    "n01768244",
    "n01770081",
    "n01770393",
    "n01773157",
    "n01773549",
    "n01773797",
    "n01774384",
    "n01774750",
    "n01775062",
    "n01776313",
    "n01784675",
    "n01795545",
    "n01796340",
    "n01797886",
    "n01798484",
    "n01806143",
    "n01806567",
    "n01807496",
    "n01817953",
    "n01818515",
    "n01819313",
    "n01820546",
    "n01824575",
    "n01828970",
    "n01829413",
    "n01833805",
    "n01843065",
    "n01843383",
    "n01847000",
    "n01855032",
    "n01855672",
    "n01860187",
    "n01871265",
    "n01872401",
    "n01873310",
    "n01877812",
    "n01882714",
    "n01883070",
    "n01910747",
    "n01914609",
    "n01917289",
    "n01924916",
    "n01930112",
    "n01943899",
    "n01944390",
    "n01945685",
    "n01950731",
    "n01955084",
    "n01968897",
    "n01978287",
    "n01978455",
    "n01980166",
    "n01981276",
    "n01983481",
    "n01984695",
    "n01985128",
    "n01986214",
    "n01990800",
    "n02002556",
    "n02002724",
    "n02006656",
    "n02007558",
    "n02009229",
    "n02009912",
    "n02011460",
    "n02012849",
    "n02013706",
    "n02017213",
    "n02018207",
    "n02018795",
    "n02025239",
    "n02027492",
    "n02028035",
    "n02033041",
    "n02037110",
    "n02051845",
    "n02056570",
    "n02058221",
    "n02066245",
    "n02071294",
    "n02074367",
    "n02077923",
    "n02085620",
    "n02085782",
    "n02085936",
    "n02086079",
    "n02086240",
    "n02086646",
    "n02086910",
    "n02087046",
    "n02087394",
    "n02088094",
    "n02088238",
    "n02088364",
    "n02088466",
    "n02088632",
    "n02089078",
    "n02089867",
    "n02089973",
    "n02090379",
    "n02090622",
    "n02090721",
    "n02091032",
    "n02091134",
    "n02091244",
    "n02091467",
    "n02091635",
    "n02091831",
    "n02092002",
    "n02092339",
    "n02093256",
    "n02093428",
    "n02093647",
    "n02093754",
    "n02093859",
    "n02093991",
    "n02094114",
    "n02094258",
    "n02094433",
    "n02095314",
    "n02095570",
    "n02095889",
    "n02096051",
    "n02096177",
    "n02096294",
    "n02096437",
    "n02096585",
    "n02097047",
    "n02097130",
    "n02097209",
    "n02097298",
    "n02097474",
    "n02097658",
    "n02098105",
    "n02098286",
    "n02098413",
    "n02099267",
    "n02099429",
    "n02099601",
    "n02099712",
    "n02099849",
    "n02100236",
    "n02100583",
    "n02100735",
    "n02100877",
    "n02101006",
    "n02101388",
    "n02101556",
    "n02102040",
    "n02102177",
    "n02102318",
    "n02102480",
    "n02102973",
    "n02104029",
    "n02104365",
    "n02105056",
    "n02105162",
    "n02105251",
    "n02105412",
    "n02105505",
    "n02105641",
    "n02105855",
    "n02106030",
    "n02106166",
    "n02106382",
    "n02106550",
    "n02106662",
    "n02107142",
    "n02107312",
    "n02107574",
    "n02107683",
    "n02107908",
    "n02108000",
    "n02108089",
    "n02108422",
    "n02108551",
    "n02108915",
    "n02109047",
    "n02109525",
    "n02109961",
    "n02110063",
    "n02110185",
    "n02110341",
    "n02110627",
    "n02110806",
    "n02110958",
    "n02111129",
    "n02111277",
    "n02111500",
    "n02111889",
    "n02112018",
    "n02112137",
    "n02112350",
    "n02112706",
    "n02113023",
    "n02113186",
    "n02113624",
    "n02113712",
    "n02113799",
    "n02113978",
    "n02114367",
    "n02114548",
    "n02114712",
    "n02114855",
    "n02115641",
    "n02115913",
    "n02116738",
    "n02117135",
    "n02119022",
    "n02119789",
    "n02120079",
    "n02120505",
    "n02123045",
    "n02123159",
    "n02123394",
    "n02123597",
    "n02124075",
    "n02125311",
    "n02127052",
    "n02128385",
    "n02128757",
    "n02128925",
    "n02129165",
    "n02129604",
    "n02130308",
    "n02132136",
    "n02133161",
    "n02134084",
    "n02134418",
    "n02137549",
    "n02138441",
]


def get_prop_raw_image_paths(num_workers, target_worker_gb):
    """Get a subset of imagenet raw image paths such that the dataset can be divided
    evenly across workers, with each receiving target_worker_gb GB of data.
    The resulting dataset size is roughly num_workers * target_worker_gb GB."""
    img_s3_root = "s3://anyscale-imagenet/ILSVRC/Data/CLS-LOC/train"
    if target_worker_gb == -1:
        # Return the entire dataset.
        return img_s3_root

    mb_per_file = 143  # averaged across 300 classes

    TARGET_NUM_DIRS = min(
        math.ceil(target_worker_gb * num_workers * 1024 / mb_per_file),
        len(IMAGENET_CLASS_IDS),
    )
    file_paths = [
        f"{img_s3_root}/{class_id}/"
        for class_id in IMAGENET_CLASS_IDS[:TARGET_NUM_DIRS]
    ]
    return file_paths


def get_prop_parquet_paths(num_workers, target_worker_gb):
    parquet_s3_dir = "s3://anyscale-imagenet/parquet"
    parquet_s3_root = f"{parquet_s3_dir}/d76458f84f2544bdaac158d1b6b842da"
    if target_worker_gb == -1:
        # Return the entire dataset.
        return parquet_s3_dir

    mb_per_file = 128
    num_files = 200
    TARGET_NUM_FILES = min(
        math.ceil(target_worker_gb * num_workers * 1024 / mb_per_file), num_files
    )
    file_paths = []
    for fi in range(num_files):
        for i in range(5):
            if not (fi in [163, 164, 174, 181, 183, 190] and i == 4):
                # for some files, they only have 4 shards instead of 5.
                file_paths.append(f"{parquet_s3_root}_{fi:06}_{i:06}.parquet")
            if len(file_paths) >= TARGET_NUM_FILES:
                break
        if len(file_paths) >= TARGET_NUM_FILES:
            break
    return file_paths

def get_test_prop_parquet_paths(num_workers, target_worker_gb):
    parquet_s3_dir = "s3://ai-ref-arch/imagenet-mini-parquet/"
    file_paths = []
    for fi in range(9):
        file_paths.append(f"{parquet_s3_dir}imagenet_{fi}.parquet")
    return file_paths


def get_mosaic_epoch_size(num_workers, target_worker_gb=10):
    if target_worker_gb == -1:
        return None
    AVG_MOSAIC_IMAGE_SIZE_BYTES = 500 * 1024  # 500KiB.
    epoch_size = math.ceil(
        target_worker_gb
        * num_workers
        * 1024
        * 1024
        * 1024
        / AVG_MOSAIC_IMAGE_SIZE_BYTES
    )
    return epoch_size
