// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <ivqML/IO.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

namespace ivqML
{
  namespace IO
  {
    namespace _ReadIDX
    {
      /**
       */
      unsigned int SwapEndianess( const unsigned int& v )
      {
        unsigned int lmost   = ( ( v & 0x000000FF ) >>  0 ) << 24;
        unsigned int lmiddle = ( ( v & 0x0000FF00 ) >>  8 ) << 16;
        unsigned int rmiddle = ( ( v & 0x00FF0000 ) >> 16 ) << 8;
        unsigned int rmost   = ( ( v & 0xFF000000 ) >> 24 );

        return(
          ( ( ( v & 0x000000FF ) >> 0 ) << 24 )
          |
          ( ( ( v & 0x0000FF00 ) >> 8 ) << 16 )
          |
          ( ( ( v & 0x00FF0000 ) >> 16 ) << 8 )
          |
          ( ( v & 0xFF000000 ) >> 24 )
          );
      }

      /**
       */
      template< class _TReal, class _TData >
      void Cast(
        Eigen::Matrix< _TReal, Eigen::Dynamic, Eigen::Dynamic >& D,
        char* B, unsigned int rows, unsigned int cols
        )
      {
        using TMatrix
          =
          Eigen::Matrix< _TData, Eigen::Dynamic, Eigen::Dynamic >;
        using TMap = Eigen::Map< TMatrix >;
        D
          =
          TMap(
            reinterpret_cast< _TData* >( B ), rows, cols
            )
          .template cast< _TReal >( );
      }
    } // end namespace
  } // end namespace
} // end namespace

// -------------------------------------------------------------------------
template< class _TReal >
void ivqML::IO::
ReadIDX(
  Eigen::Matrix< _TReal, Eigen::Dynamic, Eigen::Dynamic >& D,
  const std::string& fname
  )
{
  std::ifstream ifs( fname.c_str( ), std::ios_base::binary );
  /* TODO
     if( !ifs )
     return( false );
  */
  ifs.seekg( 0, std::ios_base::end );
  auto L = ifs.tellg( );
  ifs.seekg( 0, std::ios_base::beg );
  std::vector< char > V( L );
  char* B = V.data( );
  ifs.read( B, L );
  ifs.close( );

  // Read header
  unsigned short magic = *( reinterpret_cast< unsigned short* >( B ) );
  /* TODO
     if( magic != 0 )
     throw
  */
  unsigned char type = *( reinterpret_cast< unsigned char* >( B + 2 ) );
  unsigned char dims = *( reinterpret_cast< unsigned char* >( B + 3 ) );

  // Read sizes
  unsigned long long i = 4;
  unsigned int cols
    =
    ivqML::IO::_ReadIDX::SwapEndianess(
      *( reinterpret_cast< unsigned int* >( B + i ) )
      );
  unsigned int rows = 1;
  for( unsigned char j = 1; j < dims; ++j )
  {
    i += sizeof( unsigned int );
    rows *=
      ivqML::IO::_ReadIDX::SwapEndianess(
        *( reinterpret_cast< unsigned int* >( B + i ) )
        );
  } // end for
  i += sizeof( unsigned int );

  // Read raw data
  if( type == 0x08 )
    ivqML::IO::_ReadIDX::Cast< _TReal, unsigned char >( D, B + i, rows, cols );
  else if( type == 0x09 )
    ivqML::IO::_ReadIDX::Cast< _TReal, char >( D, B + i, rows, cols );
  else if( type == 0x0B )
    ivqML::IO::_ReadIDX::Cast< _TReal, short >( D, B + i, rows, cols );
  else if( type == 0x0C )
    ivqML::IO::_ReadIDX::Cast< _TReal, int >( D, B + i, rows, cols );
  else if( type == 0x0D )
    ivqML::IO::_ReadIDX::Cast< _TReal, float >( D, B + i, rows, cols );
  else if( type == 0x0E )
    ivqML::IO::_ReadIDX::Cast< _TReal, double >( D, B + i, rows, cols );
  /* TODO
     else
     throw
  */
}

// -------------------------------------------------------------------------
template< class _TReal >
void ivqML::IO::
ReadMNIST(
  Eigen::Matrix< _TReal, Eigen::Dynamic, Eigen::Dynamic >& Xtr,
  Eigen::Matrix< _TReal, Eigen::Dynamic, Eigen::Dynamic >& Ytr,
  Eigen::Matrix< _TReal, Eigen::Dynamic, Eigen::Dynamic >& Xte,
  Eigen::Matrix< _TReal, Eigen::Dynamic, Eigen::Dynamic >& Yte,
  const std::string& dname
  )
{
  auto fXtr = std::filesystem::path( dname ) / "train-images.idx3-ubyte";
  auto fYtr = std::filesystem::path( dname ) / "train-labels.idx1-ubyte";
  auto fXte = std::filesystem::path( dname ) / "t10k-images.idx3-ubyte";
  auto fYte = std::filesystem::path( dname ) / "t10k-labels.idx1-ubyte";

  ReadIDX( Xtr, fXtr );
  ReadIDX( Ytr, fYtr );
  ReadIDX( Xte, fXte );
  ReadIDX( Yte, fYte );
}

// -------------------------------------------------------------------------
#define ivqML_IO_ReadIDX( t )                            \
  template ivqML_EXPORT void ivqML::IO::ReadIDX< t >(    \
    Eigen::Matrix< t, Eigen::Dynamic, Eigen::Dynamic >&, \
    const std::string&                                   \
    );                                                   \
  template ivqML_EXPORT void ivqML::IO::ReadMNIST< t >(  \
    Eigen::Matrix< t, Eigen::Dynamic, Eigen::Dynamic >&, \
    Eigen::Matrix< t, Eigen::Dynamic, Eigen::Dynamic >&, \
    Eigen::Matrix< t, Eigen::Dynamic, Eigen::Dynamic >&, \
    Eigen::Matrix< t, Eigen::Dynamic, Eigen::Dynamic >&, \
    const std::string&                                   \
    )

ivqML_IO_ReadIDX( char );
ivqML_IO_ReadIDX( short );
ivqML_IO_ReadIDX( int );
ivqML_IO_ReadIDX( long );
ivqML_IO_ReadIDX( long long );
ivqML_IO_ReadIDX( signed char );
ivqML_IO_ReadIDX( unsigned char );
ivqML_IO_ReadIDX( unsigned short );
ivqML_IO_ReadIDX( unsigned int );
ivqML_IO_ReadIDX( unsigned long );
ivqML_IO_ReadIDX( unsigned long long );
ivqML_IO_ReadIDX( float );
ivqML_IO_ReadIDX( double );
ivqML_IO_ReadIDX( long double );

// eof - $RCSfile$
