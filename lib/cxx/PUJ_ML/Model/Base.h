// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Model__Base__h__
#define __PUJ_ML__Model__Base__h__

#include <iostream>
#include <vector>
#include <Eigen/Core>

namespace PUJ_ML
{
  namespace Model
  {
    /**
     */
    template< class _D, class _R >
    class Base
    {
    public:
      using Self = Base;
      using Derived = _D;

      using TReal = _R;
      using TMatrix = Eigen::Matrix< _R, Eigen::Dynamic, Eigen::Dynamic >;
      using TCol = Eigen::Matrix< _R, Eigen::Dynamic, 1 >;
      using TRow = Eigen::Matrix< _R, 1, Eigen::Dynamic >;

    public:
      Base( const unsigned long long& n = 1 );
      virtual ~Base( ) = default;

      unsigned long long number_of_parameters( ) const;
      unsigned long long number_of_inputs( ) const;
      void init( const unsigned long long& n );

      TReal& operator()( const unsigned long long& i );
      const TReal& operator()( const unsigned long long& i ) const;

      template< class _Y, class _X >
      void threshold(
        Eigen::EigenBase< _Y >& Y, const Eigen::EigenBase< _X >& X
        ) const;

      void move_parameters(
        const std::vector< TReal >& dir,
        const TReal& coeff = TReal( 1 )
        );

    protected:
      void _to_stream( std::ostream& o ) const;

    protected:
      std::vector< TReal > m_P;

    public:
      friend std::ostream& operator<<( std::ostream& o, const Self& m )
        {
          m._to_stream( o );
          return( o );
        }
    };
  } // end namespace
} // end namespace

#include <PUJ_ML/Model/Base.hxx>

#endif // __PUJ_ML__Model__Base__h__

// eof - $RCSfile$
